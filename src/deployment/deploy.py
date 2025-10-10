from datetime import datetime
import json
from uuid import uuid4
from mlflow import MlflowClient, set_tracking_uri
from utils.logger import logger
from src.mlops.governance import Governance
from src.deployment.traffic_router import TrafficRouter
from utils.results import Result, Status
from utils.audit_logger import audit_log


class DeploymentManager:
    """Manager eployment: promotion, routing, rollback, and governance."""

    def __init__(self, config):
        self.config = config
        uri = config.get("mlflow_config.remote_server_uri") or config.get(
            "mlflow_config.mlflow_tracking_uri"
        )
        if not uri:
            raise ValueError("No MLflow tracking URI found in config")
        set_tracking_uri(uri)
        self.client = MlflowClient(tracking_uri=uri)
        self.governance = Governance(config)
        self.traffic_router = TrafficRouter(config)
        self.author = config.get("project.author", "unknown")

    def get_latest_model(self, model_name: str, aliases=("staging", "production")):
        """Retrieve the latest model version across specified aliases."""
        latest, latest_version = None, -1

        for alias in aliases:
            try:
                mv = self.client.get_model_version_by_alias(model_name, alias)
                if mv and int(mv.version) > latest_version:
                    latest = mv
                    latest_version = int(mv.version)
                    logger.info(
                        f"Found model {model_name} v{mv.version} with alias '{alias}'"
                    )
            except Exception as e:
                logger.warning(f"Error checking alias '{alias}': {str(e)}")

        if latest:
            return {
                "model_name": model_name,
                "version": int(latest.version),
                "alias": latest.aliases,
                "run_id": latest.run_id,
                "creation_timestamp": latest.creation_timestamp,
            }
        return None
    
    
    def deploy(self, exp_info: dict) -> Result:
        """Deployment pipeline workflow"""
        model_name = exp_info["model_name"]
        candidate_version = int(exp_info.get("model_version"))
        candidate_run_id = exp_info.get("run_id")
        strategy = self.governance.default_strategy

        if not candidate_run_id:
            return Result(Status.FAILED, "Missing run_id in experiment info")

        current_model_info = None
        previous_model = {}

        # Audit: Start deployment
        audit_log(
            "DeploymentStart",
            {
                "model": model_name,
                "version": candidate_version,
                "run_id": candidate_run_id,
                "strategy": strategy,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        try:
            # Step 1: Get current production model
            current_model_info = self.get_latest_model(
                model_name, aliases=("production",)
            )

            # Step 2: Validate Metric, Governance, Strategy
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")
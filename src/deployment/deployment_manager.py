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

    def get_previous_model_info(self, model_name: str, current_version: int) -> dict:
        """Retrieve detailed information about the previous model version."""
        try:
            current_version = int(current_version)
            versions = self.client.search_model_versions(f"name='{model_name}'")

            if not versions:
                return {}

            sorted_versions = sorted(
                versions, key=lambda x: int(x.version), reverse=True
            )

            for mv in sorted_versions:
                v_num = int(mv.version)
                if v_num >= current_version:
                    continue

                model_version = self.client.get_model_version(model_name, str(v_num))
                tags = model_version.tags or {}

                # Fetch metrics
                metrics = {}
                try:
                    run = self.client.get_run(mv.run_id)
                    metrics = dict(run.data.metrics)
                except Exception as e:
                    logger.warning(f"Failed to fetch metrics for run {mv.run_id}: {e}")

                previous_model_info = {
                    "model_name": model_name,
                    "version": v_num,
                    "run_id": mv.run_id,
                    "alias": mv.aliases or [],
                    "routing_config": json.loads(tags.get("routing_target", "{}")),
                    "deployment_timestamp": tags.get("deployment_timestamp", ""),
                    "metrics": metrics,
                    "evaluation_decision": json.loads(
                        tags.get("evaluation_decision", "{}")
                    ),
                    "strategy_decision": json.loads(
                        tags.get("strategy_decision", "{}")
                    ),
                }
                logger.info(f"Found previous model: v{v_num}")
                return previous_model_info

            return {}
        except Exception as e:
            logger.error(f"Error retrieving previous model info: {e}")
            return {}

    def validate_model_metrics(self, run_id: str) -> dict:
        """Retrieve and validate model metrics from MLflow run."""
        try:
            run = self.client.get_run(run_id)
            return dict(run.data.metrics)
        except Exception as e:
            logger.error(f"Failed to retrieve metrics for run {run_id}: {str(e)}")
            return {}

    def promote_model(
        self, model_name: str, version: int, target_alias: str = "production"
    ) -> bool:
        """Promote a model version to the specified alias."""
        try:
            # Remove existing alias if present
            try:
                current_prod = self.client.get_model_version_by_alias(
                    model_name, target_alias
                )
                if current_prod:
                    self.client.delete_registered_model_alias(model_name, target_alias)
                    logger.info(
                        f"Removed existing '{target_alias}' alias from v{current_prod.version}"
                    )
            except Exception:
                pass

            # Set new alias
            self.client.set_registered_model_alias(
                model_name, target_alias, str(version)
            )
            logger.info(f"Promoted {model_name} v{version} → '{target_alias}'")
            return True
        except Exception as e:
            logger.error(f"Failed promotion {model_name} v{version}: {str(e)}")
            return False

    def get_rollback_candidate(self, model_name: str) -> dict:
        """Identify the most recent valid rollback candidate."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return {}

            sorted_versions = sorted(
                versions, key=lambda x: int(x.version), reverse=True
            )
            prod_version = None

            # Find current production version
            for mv in sorted_versions:
                aliases = mv.aliases or []
                if "production" in aliases and not prod_version:
                    prod_version = int(mv.version)
                    continue

                # Return first version before production
                if prod_version and int(mv.version) < prod_version:
                    return {
                        "version": int(mv.version),
                        "run_id": mv.run_id,
                        "creation_timestamp": mv.creation_timestamp,
                        "source": mv.source,
                    }
            return {}
        except Exception as e:
            logger.error(f"Error finding rollback candidate: {e}")
            return {}

    def perform_rollback(
        self,
        model_name: str,
        target_version: int = None,
        reason: str = "Automatic rollback due to failure",
    ) -> Result:
        """Perform rollback to a previous stable model version."""
        # Determine rollback target
        if target_version is None:
            rollback_candidate = self.get_rollback_candidate(model_name)
            if not rollback_candidate:
                logger.warning("No rollback candidate found.")
                return Result(Status.FAILED, "No rollback candidate available")
            version = rollback_candidate["version"]
            run_id = rollback_candidate["run_id"]
        else:
            try:
                mv = self.client.get_model_version(model_name, str(target_version))
                version = target_version
                run_id = mv.run_id
            except Exception as e:
                logger.error(f"Target version {target_version} not found: {e}")
                return Result(
                    Status.FAILED, f"Target version {target_version} not found"
                )

        # Governance validation of rollback model
        metrics = self.validate_model_metrics(run_id)
        evaluation_decision = self.governance.policy_evaluate(metrics)

        if not evaluation_decision["approved"]:
            logger.warning(
                f"Rollback candidate v{version} failed governance validation: "
                f"{evaluation_decision['reasons']}"
            )
            audit_log(
                "RollbackBlocked",
                {
                    "model": model_name,
                    "candidate_version": version,
                    "reasons": evaluation_decision["reasons"],
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            return Result(
                Status.FAILED,
                "Rollback blocked: candidate did not meet governance policy",
                evaluation_decision,
            )

        # Execute rollback
        try:
            rollback_time = datetime.utcnow().isoformat()

            # Set production alias
            self.client.set_registered_model_alias(
                model_name, "production", str(version)
            )

            # Add rollback metadata tags
            rollback_tags = {
                "rollback_from": "automatic" if target_version is None else "manual",
                "rollback_reason": reason,
                "rollback_timestamp": rollback_time,
                "rollback_by": self.author,
            }

            for key, value in rollback_tags.items():
                self.client.set_model_version_tag(model_name, str(version), key, value)

            audit_log(
                "RollbackSuccess",
                {
                    "model": model_name,
                    "rolled_back_to": version,
                    "reason": reason,
                    "timestamp": rollback_time,
                },
            )

            logger.info(f"Rollback executed: {model_name} → v{version}")

            return Result(
                Status.SUCCESS,
                f"Rollback successful to version {version}",
                {
                    "model_name": model_name,
                    "version": version,
                    "run_id": run_id,
                    "rolled_back_at": rollback_time,
                    "reason": reason,
                },
            )
        except Exception as e:
            audit_log(
                "RollbackFailed",
                {
                    "model": model_name,
                    "target_version": version,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            logger.error(f"Rollback execution failed: {e}")
            return Result(Status.FAILED, f"Rollback failed: {e}")

    def set_deployment_tags(
        self, model_name: str, version: int, deployment_info: dict
    ) -> dict:
        """Set deployment tags on model."""
        deployment_id = str(uuid4())

        tags = {
            "deployment_id": deployment_id,
            "deployment_status": deployment_info.get("status", "active"),
            "deployment_alias": deployment_info.get("alias", "production"),
            "deployed_strategy": deployment_info.get("strategy", "direct"),
            "routing_target": json.dumps(deployment_info.get("routing_config", {})),
            "deployment_timestamp": datetime.utcnow().isoformat(),
            "deployed_by": self.author,
            "evaluation_decision": json.dumps(
                deployment_info.get("evaluation_decision", {})
            ),
            "strategy_decision": json.dumps(
                deployment_info.get("strategy_decision", {})
            ),
            "previous_model": json.dumps(deployment_info.get("previous_model", {})),
        }

        for key, value in tags.items():
            try:
                self.client.set_model_version_tag(
                    model_name, str(version), key, str(value)
                )
            except Exception as e:
                logger.warning(f"Failed to set tag {key}: {str(e)}")

        return tags

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

            # Step 2: Validate candidate metrics
            candidate_metrics = self.validate_model_metrics(candidate_run_id)
            if not candidate_metrics:
                return Result(Status.FAILED, "No candidate metrics found")

            # Step 3: Governance evaluation
            evaluation_decision = self.governance.policy_evaluate(candidate_metrics)
            if not evaluation_decision["approved"]:
                audit_log(
                    "GovernanceBlocked",
                    {
                        "model": model_name,
                        "version": candidate_version,
                        "decision": evaluation_decision,
                    },
                )
                return Result(
                    Status.FAILED,
                    "Blocked by governance evaluation",
                    evaluation_decision,
                )

            # Step 4: Strategy validation
            strategy_decision = self.governance.policy_deployment(strategy)
            if not strategy_decision["approved"]:
                audit_log(
                    "StrategyBlocked",
                    {
                        "model": model_name,
                        "version": candidate_version,
                        "strategy": strategy,
                    },
                )
                return Result(
                    Status.FAILED,
                    "Blocked by governance strategy",
                    strategy_decision,
                )

            # Step 5: Promote model to production
            if (
                not current_model_info
                or current_model_info["version"] != candidate_version
            ):
                if not self.promote_model(model_name, candidate_version, "production"):
                    return Result(Status.FAILED, "Promotion failed")

            # Step 6: Configure traffic routing
            routing_config = self.traffic_router.get_routing_config(strategy)

            # Step 7: Retrieve previous model info
            if current_model_info:
                previous_model = self.get_previous_model_info(
                    model_name, candidate_version
                )
                logger.info(f"Previous model info: {previous_model}")

            # Step 8: Set deployment metadata
            deployment_info = {
                "status": "active",
                "alias": "production",
                "strategy": strategy,
                "routing_config": routing_config,
                "evaluation_decision": evaluation_decision,
                "strategy_decision": strategy_decision,
                "previous_model": previous_model,
            }
            tags = self.set_deployment_tags(
                model_name, candidate_version, deployment_info
            )

            # Step 9: Prepare deployment result
            deployment_data = {
                "model_name": model_name,
                "version": candidate_version,
                "run_id": candidate_run_id,
                "metrics": candidate_metrics,
                "evaluation_decision": evaluation_decision,
                "strategy_decision": strategy_decision,
                "routing_config": routing_config,
                "previous_model": previous_model,
                "timestamp": datetime.utcnow().isoformat(),
                "tags": tags,
            }

            audit_log(
                "DeploymentSuccess",
                {
                    "model": model_name,
                    "version": candidate_version,
                    "strategy": strategy,
                    "routing": routing_config,
                },
            )

            return Result(
                Status.SUCCESS, "Deployment completed successfully", deployment_data
            )

        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")

            # Automatic rollback on failure
            if current_model_info:
                logger.info(f"Attempting automatic rollback due to deployment failure")
                rollback_result = self.perform_rollback(
                    model_name, reason=f"Deployment failure: {str(e)}"
                )

                if rollback_result.status == Status.SUCCESS:
                    logger.info(
                        "Rollback completed successfully after deployment failure"
                    )
                else:
                    logger.error(f"Rollback attempt failed: {rollback_result.message}")

            audit_log(
                "DeploymentFailed",
                {
                    "model": model_name,
                    "version": candidate_version,
                    "error": str(e),
                    "rollback_attempted": current_model_info is not None,
                },
            )

            return Result(Status.FAILED, f"Deployment pipeline error: {str(e)}")

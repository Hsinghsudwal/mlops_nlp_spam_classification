from utils.results import Status
from src.experiment.experiment_tracker import MLflowTracker
from src.experiment.experiment_manager import ExperimentManager


class ExperimentPipeline:
    """MLflow experiment."""

    def __init__(self, config, artifact_manager, user_input=None):
        self.config = config
        self.user_input = user_input
        self.artifact_manager = artifact_manager
        self.manager = ExperimentManager(self.config)
        self.policy_promote = self.config.get("policy.allow_promotion", False)

    def run(self, pipeline_results):
        # Register + tracker
        tracker = MLflowTracker(self.config)
        tracker_result = tracker.track_experiment(
            pipeline_results,
            pipeline_id=pipeline_results["pipeline_metadata"]["pipeline_id"],
        )
        if tracker_result.status != Status.SUCCESS:
            return {
                "model_name": None,
                "model_version": None,
                "status": tracker_result.status,
            }

        model_info = tracker_result.data.get("model_info")
        print("registered_info:'", model_info)
        model_name = model_info["model_name"]
        model_version = model_info["model_version"]
        run_id = model_info["run_id"]

        exp_artifact = {
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
            "status": Status.SUCCESS.name,
        }

        pipeline_id = pipeline_results["pipeline_metadata"]["pipeline_id"]

        self.artifact_manager.save(
            artifact=exp_artifact,
            subdir="registry",
            name="latest_experiment.json",
            pipeline_id=pipeline_id,
        )

        # Stage
        stage_result = self.manager.stage_model(model_name, model_version)
        if stage_result.status != Status.SUCCESS:
            return {
                "model_name": model_name,
                "model_version": model_version,
                "status": stage_result.status,
            }
        print(f"Staged {model_name} v{model_version}")

        # Decide promotion
        promote = (
            self.user_input.lower() == "y" if self.user_input else self.policy_promote
        )
        if not promote:
            print(f"Skipping promotion for {model_name} v{model_version}")
            return {
                "model_name": model_name,
                "model_version": model_version,
                "status": Status.SUCCESS,
            }

        prod_result = self.manager.safe_promote_to_production(model_name)
        print(prod_result.message)

        return {
            "model_name": model_name,
            "model_version": model_version,
            "status": prod_result.status,
        }

import argparse
import sys
import os
import subprocess
from utils.artifact_manager import ArtifactManager
from utils.config_manager import ConfigManager
from integration.storage import ArtifactStoreFactory
from pipelines.training_pipeline import TrainingPipeline
from pipelines.experiment_pipeline import ExperimentPipeline
from pipelines.deployment_pipeline import DeploymentPipeline
from pipelines.monitoring_pipeline import MonitorPipeline

# from pipelines.orchestrating_pipeline import OrchestratingPipeline


def train_pipeline(config, artifact_manager, data_path, user_input):
    """Run training + experiment registration. Promotion only affects experiment stage."""
    training_pipeline = TrainingPipeline(
        data_path=data_path, config=config, artifact_manager=artifact_manager
    )
    train_results = training_pipeline.run()
    print("Training results keys:", train_results.keys())

    if train_results.get("pipeline_metadata", {}).get("status") == "completed":
        exp_pipeline = ExperimentPipeline(
            config=config, artifact_manager=artifact_manager, user_input=user_input
        )
        exp_info = exp_pipeline.run(train_results)
        print("Experiment registered:", exp_info)
        return exp_info

    return None


def deploy_pipeline(config, artifact_manager):
    """Deploy model based on latest experiment info."""
    exp_info = artifact_manager.load(subdir="registry", name="latest_experiment.json")
    if not exp_info:
        raise ValueError("No experiment found to deploy")
    deployment_pipeline = DeploymentPipeline(
        config=config, artifact_manager=artifact_manager
    )
    deploy_result = deployment_pipeline.run(exp_info)
    print("Deployment result:", deploy_result)
    return deploy_result


def serve_pipeline():
    """Serve model using Streamlit or other server."""
    print("Starting model serving...")
    subprocess.run([sys.executable, "app.py"], check=True)


def monitor_pipeline(config, artifact_manager):
    """Standalone monitoring"""
    monitor_pipeline = MonitorPipeline(config=config, artifact_manager=artifact_manager)
    monitor_result = monitor_pipeline.run()
    print("Monitor result:", monitor_result)
    return monitor_result


def run_orchestrated_pipeline():
    """Run orchestrated autonomous system."""
    print("Running orchestrated Autonomus System...")
    # orchestrator = OrchestratingPipeline()
    # return orchestrator.run()
    print("Orchestration not implemented yet.")


def main():
    parser = argparse.ArgumentParser(description="Run Autonomous ML Pipeline")

    # Storage options (mutually exclusive)
    storage_group = parser.add_mutually_exclusive_group(required=True)
    storage_group.add_argument(
        "--local", action="store_true", help="Use local file storage"
    )
    storage_group.add_argument(
        "--cloud", action="store_true", help="Use AWS cloud storage"
    )
    storage_group.add_argument(
        "--localstack", action="store_true", help="Use LocalStack for testing"
    )

    # Pipeline mode
    parser.add_argument(
        "--data", default="data/SMSSpamData.tsv", help="Path to dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "deploy", "serve", "monitor", "orchestrated"],
        default="train",
    )
    parser.add_argument(
        "--prod", choices=["y", "n"], default="n", help="Promote model to production?"
    )

    args = parser.parse_args()

    # Determine storage mode
    storage_mode = "local" if args.local else "cloud" if args.cloud else "localstack"

    # Load config
    config_path = "config/config.yaml"
    config = ConfigManager.load_file(config_path)
    config.set("storage.mode", storage_mode)

    # Artifact manager
    artifact_manager = ArtifactStoreFactory.create_store(config)

    try:
        if args.mode == "train":
            train_pipeline(
                config, artifact_manager, data_path=args.data, user_input=args.prod
            )
        elif args.mode == "deploy":
            deploy_pipeline(config, artifact_manager)
        elif args.mode == "serve":
            serve_pipeline()
        elif args.mode == "monitor":
            monitor_pipeline(config, artifact_manager)
        elif args.mode == "orchestrated":
            run_orchestrated_pipeline()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    """1. Train
    2. deployment
    3. serve
    4. monitor
    5. orchestrate"""
    main()

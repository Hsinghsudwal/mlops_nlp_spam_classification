import argparse
import sys
import os
import subprocess
from pipelines.training_pipeline import TrainingPipeline
from pipelines.experiment_pipeline import ExperimentPipeline
from pipelines.deployment_pipeline import DeploymentPipeline

# from pipelines.orchestrating_pipeline import OrchestratingPipeline
import app


def run_ml_cycle(config_path: str, prod: bool = False):
    pass
def main():

    parser = argparse.ArgumentParser(description="Run Pipeline for Autonomus")

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

    # Data Pipeline mode
    parser.add_argument(
        "--data", default="data/SMSSpamData.tsv" or None, help="Path to dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["training", "deploy", "serve", "orchestrated"],
        default="training",
        help="Pipeline execution mode",
    )
    
    parser.add_argument(
        "--prod",
        choices=["y", "n"],
        default="n",
        help="Promote model",
    )

    args = parser.parse_args()

    # Determine storage mode
    if args.local:
        storage_mode = "local"
    elif args.cloud:
        storage_mode = "cloud"
    elif args.localstack:
        storage_mode = "localstack"
    else:
        print("Error: Please specify storage mode (--local, --cloud, or --localstack)")
        sys.exit(1)

    # Load and Validate config file
    config_path = "config/local.yaml"

    try:
        if args.mode == "training":
            print("Starting training mode...")
            pipeline = TrainingPipeline(
                data_path=args.data, config_path=config_path, storage_mode=storage_mode
            )
            train_results = pipeline.run()
            # print("Training pipeline completed successfully!")
            print(f"Results: {train_results.keys()}")  # train_results.keys()
            # print(train_results)
            if train_results["pipeline_metadata"].get("status") == "completed":
                print("passed")
                # sys.exit(1)
                exp_mlflow = ExperimentPipeline(config_path, user_input=args.prod)
                exp_info = exp_mlflow.run(train_results)
                print(exp_info)

        # Handle deployment if requested
        elif args.mode == "deploy":
            DeploymentPipeline(config_path).run()
            

        elif args.mode == "serve":
            print("Serving model with Streamlit...")
            subprocess.run(["streamlit", "run", "app.py"], check=True)

        elif args.mode == "orchestrated":
            print("Orchestrated Autonomus System.")
            # TODO: Implement orchestrated pipeline
            # orchestrator = OrchestratingPipeline()
            # orchest_results = orchestrator.run()

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

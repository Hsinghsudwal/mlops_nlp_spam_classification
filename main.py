import argparse
import sys
import os
from pipelines.training_pipeline import TrainingPipeline
# from pipelines.orchestrating_pipeline import OrchestratingPipeline


def main():

    
    parser = argparse.ArgumentParser(description="Run MLOps Pipeline for Ham/Spam Classification")
    
    # Storage options (mutually exclusive)
    storage_group = parser.add_mutually_exclusive_group(required=True)
    storage_group.add_argument("--local", action="store_true", help="Use local file storage")
    storage_group.add_argument("--cloud", action="store_true", help="Use AWS cloud storage")  
    storage_group.add_argument("--localstack", action="store_true", help="Use LocalStack for testing")
    
    # Data path
    parser.add_argument("--data", default="data/SMSSpamData.tsv", help="Path to spam/ham dataset")
    
    # Pipeline mode
    parser.add_argument("--mode",
                        choices=["training", "serving", "orchestrated"],
                        default="training",
                        help="Pipeline execution mode")
    

    parser.add_argument("--deploy", action="store_true", help="Deploy model after training")
    parser.add_argument("--serve", action="store_true", help="Start model server")
    
    args = parser.parse_args()
    
    # Validate data path
    # if not os.path.exists(args.data):
    #     print(f"Error: Data file not found: {args.data}")
    #     sys.exit(1)
    
    # Determine config file
    if args.local:
        config_file = "config/local.yaml"
    elif args.cloud:
        config_file = "config/cloud.yaml"
    elif args.localstack:
        config_file = "config/localstack.yaml"
    else:
        print("Error: Please specify storage mode (--local, --cloud, or --localstack)")
        sys.exit(1)
    
    # Validate config file
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    try:
        if args.mode == "training":
            print("Starting training mode...")
            
            # Run training pipeline
            pipeline = TrainingPipeline(data_path=args.data, config_file=config_file)
            train_results = pipeline.run()
            
            print("Training pipeline completed successfully!")
            print(f"Results: {train_results}")
            
            # Handle deployment if requested
            if args.deploy:
                print("Deployment requested but not implemented yet.")
                # TODO: Implement deployment logic
                
        elif args.mode == "serving":
            print("Serving mode not implemented yet.")
            # TODO: Implement serving logic
            
        elif args.mode == "orchestrated":
            print("Orchestrated mode not implemented yet.")
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

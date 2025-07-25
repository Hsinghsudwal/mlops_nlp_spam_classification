import argparse
from pipelines.training_pipeline import TrainingPipeline
from pipelines.orchestrating_pipeline import OrchestratingPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Agentic MLOps Pipeline for Ham/Spam Classification")
    parser.add_argument("--local", action="store_true", help="Use local file storage")
    parser.add_argument("--cloud", action="store_true", help="Use AWS cloud storage")
    parser.add_argument("--localstack", action="store_true", help="Use LocalStack for testing")
    
    parser.add_argument("--data", 
                       default="data/SMSSpamData.csv", 
                       help="Path to spam/ham dataset")
    
    parser.add_argument("--mode", 
                       choices=["training", "serving", "orchestrated"], 
                       default="training", 
                       help="Pipeline execution mode")
    
    parser.add_argument("--deploy", action="store_true", help="Deploy model after training")
    parser.add_argument("--serve", action="store_true", help="Start model server")
    
    args = parser.parse_args()
    
    # Determine config file
    if args.local:
        config_file = "config/local.yaml"
    elif args.cloud:
        config_file = "config/cloud.yaml"
    elif args.localstack:
        config_file = "config/localstack.yaml"
    else:
        print("Please specify --local, --cloud, or --localstack")
        return
    
    
    if args.mode == "training":
        # Run training pipeline
        pipeline = TrainingPipeline(data_path=args.data, config_file=config_file)
        train_results = pipeline.run()
        
    elif args.mode == "orchestrated":
        orchestrator = OrchestratingPipeline()
        orchest_results = orchestrator.run()

import datetime
import uuid
import traceback

from utils.config_manager import ConfigManager
from integration.storage import ArtifactStoreFactory
from utils.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

class TrainingPipeline:
    def __init__(self, data_path: str, config_file: str):
        self.config = ConfigManager.load_file(config_file)
        self.artifact_manager = ArtifactStoreFactory.create_store(self.config)
        self.data_path = data_path or self.config.get("base.data_path")
        self.pipeline_id = str(uuid.uuid4())

        # Initialize components
        # self.data_ingestion = DataIngestion(self.data_path)
        # self.data_transformation = DataTransformation(self.config)
        # self.model_trainer = ModelTrainer(self.config)
        # self.model_evaluation = ModelEvaluation(self.config)
    
    def run(self, pipeline_id: str = None):
        """Execute the complete ML pipeline"""
            
        pipeline_id = pipeline_id or self.pipeline_id
        
        pipeline_name = self.config.get("base.pipeline_name", "ML Pipeline")
        start_time = datetime.datetime.now()
        logger.info(f"Starting pipeline '{pipeline_name}' with ID: {pipeline_id}")
        print(f"Pipeline name: {pipeline_name}")
        
        try:
            # ML Pipeline
            results = {}
            
            # Data Ingestion
            logger.info("======== Data Ingestion ========")
            data_ingestion = DataIngestion(self.data_path)
            
            ingestion_result = data_ingestion.data_ingestion(
                results=results,
                config=self.config,  
                artifact_store=self.artifact_manager,
                pipeline_id=pipeline_id,
            )
            results.update(ingestion_result)
            
            logger.info("======== Data Transformation ========")
            transformer = DataTransformation(self.config)
            transformation_result = transformer.data_transformation(
                train_data=results.get("train_data"),
                test_data=results.get("test_data"), 
                config=self.config,
                artifact_store=self.artifact_manager,
                pipeline_id=pipeline_id
            )
            results.update(transformation_result)

            # Log completion
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            logger.info(f"Training pipeline completed in {duration}")
            
            # Add pipeline metadata to results
            results["pipeline_metadata"] = {
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "status": "completed"
            }
            
            return results
        
        except Exception as e:
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            logger.error(f"Pipeline failed after {duration}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return error information
            error_result = {
                "pipeline_metadata": {
                    "pipeline_id": pipeline_id,
                    "pipeline_name": pipeline_name,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration.total_seconds(),
                    "status": "failed",
                    "error": str(e)
                }
            }
            
            raise Exception(f"Pipeline {pipeline_id} failed: {str(e)}") from e

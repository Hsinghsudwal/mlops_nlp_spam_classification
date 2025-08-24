import datetime
import uuid
import traceback

from utils.logger import logger
from utils.config_manager import ConfigManager
from integration.storage import ArtifactStoreFactory
from utils.results import Status

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainingPipeline:
    def __init__(self, data_path: str, config_path: str, storage_mode: str):
        self.config = ConfigManager.load_file(config_path)
        self.config.set("storage.mode", storage_mode)

        self.artifact_manager = ArtifactStoreFactory.create_store(self.config)
        self.data_path = data_path or self.config.get("base.data_path")
        self.storage_mode = storage_mode
        self.pipeline_id = str(uuid.uuid4())

    def run(self, pipeline_id: str = None):
        pipeline_id = pipeline_id or self.pipeline_id
        pipeline_name = self.config.get("base.pipeline_name", "ML Pipeline")
        start_time = datetime.datetime.now()

        logger.info(f"Starting pipeline '{pipeline_name}' with ID: {pipeline_id}")
        # print(f"Pipeline name: {pipeline_name}")
        # print(f"Storage mode: {self.storage_mode}")
        # print(f"Data path: {self.data_path}")

        node = {
            "ingeste": DataIngestion(self.data_path),
            "transformer": DataTransformation(),
            "trainer": ModelTrainer(),
            "evaluate": ModelEvaluation(),
        }

        try:
            # ML Pipeline
            pipeline_results = {}

            logger.info("======== Data Ingestion ========")
            ingestion_results = node["ingeste"].run(
                results=pipeline_results,
                config=self.config,
                artifact_store=self.artifact_manager,
                pipeline_id=pipeline_id,
            )
            if ingestion_results.status != Status.SUCCESS:
                raise Exception(f"Ingestion failed: {ingestion_results.message}")

            pipeline_results["ingeste"] = ingestion_results
            logger.info("Data ingestion completed.")

            logger.info("======== Data Transformation ========")
            transformer_results = node["transformer"].run(
                results=pipeline_results,
                config=self.config,
                artifact_store=self.artifact_manager,
                pipeline_id=pipeline_id,
            )
            if transformer_results.status != Status.SUCCESS:
                raise Exception(
                    f"Data transformer failed: {transformer_results.message}"
                )

            pipeline_results["transformer"] = transformer_results
            logger.info("Data Transformation completed.")

            logger.info("======== Model Trainer ========")
            trainer_results = node["trainer"].run(
                results=pipeline_results,
                config=self.config,
                artifact_store=self.artifact_manager,
                pipeline_id=pipeline_id,
            )
            if trainer_results.status != Status.SUCCESS:
                raise Exception(f"Model trainer failed: {trainer_results.message}")

            pipeline_results["trainer"] = trainer_results
            logger.info("Model Trainer completed.")

            logger.info("======== Model Evaluation ========")
            evaluate_results = node["evaluate"].run(
                results=pipeline_results,
                config=self.config,
                artifact_store=self.artifact_manager,
                pipeline_id=pipeline_id,
            )
            if evaluate_results.status != Status.SUCCESS:
                raise Exception(f"Model evaluation failed: {evaluate_results.message}")

            pipeline_results["evaluate"] = evaluate_results
            logger.info("Model evaluation completed.")

            # Log completion
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            logger.info(f"Training pipeline completed in {duration}")

            # Add pipeline metadata to results
            pipeline_results["pipeline_metadata"] = {
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "storage_mode": self.config.get("storage.mode"),
                "status": "completed",
            }
            # logger.info(f"pipeline_results: {pipeline_results}")
            return pipeline_results

        except Exception as e:
            end_time = datetime.datetime.now()
            duration = end_time - start_time

            logger.error(f"Pipeline failed after {duration}: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Pipeline {pipeline_id} failed: {str(e)}") from e

from utils.logger import logger
from utils.results import Status
import uuid
from src.monitoring.monitoring import MLMonitor


class MonitorPipeline:

    def __init__(self, config, artifact_manager):
        self.config = config
        self.artifact_manager = artifact_manager

    def run(self):
        """Run the monitoring pipeline."""
        logger.info("=== Starting Monitoring====")

        monitor = MLMonitor()
        result = monitor.run_monitoring_cycle(
            config=self.config, artifact_store=self.artifact_manager
        )
        # print(result)

        logger.info("Monitoring completed")
        return {
            "result": result,
            "status": Status.SUCCESS,
        }

from utils.logger import logger
from utils.results import Status
import uuid
from src.deployment.deployment_manager import DeploymentManager


class DeploymentPipeline:
    """Deployment manager."""

    def __init__(self, config, artifact_manager):
        self.config = config
        self.artifact_manager = artifact_manager

    def run(self, exp_info):
        """Run the deployment pipeline."""
        logger.info("=== Starting Deployment====")
        deploy = DeploymentManager(self.config)
        result = deploy.deployment_manager(exp_info)
        # print(result)
        
        self.artifact_manager.save(
            artifact=result.__dict__,
            subdir="registry",
            name="latest_deployment.json",
            pipeline_id= str(uuid.uuid4()),
        )
        logger.info("Deployment completed")
        return {
            "result": result,
            "status": Status.SUCCESS,
        }

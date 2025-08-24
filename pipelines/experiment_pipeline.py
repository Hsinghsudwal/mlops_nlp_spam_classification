from utils.results import Status
from utils.config_manager import ConfigManager
from src.experiment.mlflow_tracker import MLflowTracker
from src.experiment.stage_model import StageModel
from src.experiment.prod_model import ProdModel


class ExperimentPipeline:
    """MLflow experiment."""

    def __init__(self, config_path: str):
        self.config_path = config_path


    def run(self, results):
        # Register + tracker
        ml_register = MLflowTracker(self.config_path)
        ml_register_result = ml_register.track_experiment(
            results, pipeline_id=results["pipeline_metadata"]["pipeline_id"]
        )
        
        registered_model= ml_register_result.data.get("registered_model")
        print("registered_model:'", registered_model)
        modelname= registered_model["model_name"]
        modelversion= registered_model["model_version"]

        # Staging
        stagemodel = StageModel(self.config_path)
        stage_result = stagemodel.stage_model(modelname,modelversion)
        if stage_result.status != Status.SUCCESS:
            return stage_result
        print(f"Staged {modelname} v{modelversion}")
        
        # Production
        prodmodel = ProdModel(self.config_path)
        prod_result = prodmodel.promote_to_production(modelname)
        if prod_result.status == Status.SUCCESS:
            print(f"Promoted {modelname} v{modelversion} to Production")
        else:
            print(f"Promotion failed for {modelname} v{modelversion}")
    
        return prod_result

import json
import os
from datetime import datetime


def save_metadata(metadata: dict, output_path: str = "data/pipeline_metadata.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata["timestamp"] = datetime.utcnow().isoformat()

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
        
def metadata_deploy(metadata: dict, output_path: str = "data/deploy_metadata.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata["timestamp"] = datetime.utcnow().isoformat()

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

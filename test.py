import os
from pathlib import Path

# Define full directory structure and placeholder files
structure = {
    "fraud_detection_system": {
        "config": [
            "__init__.py",
            "settings.py",
            "model_config.yaml"
        ],
        "data": [
            "__init__.py",
            "ingestion.py",
            "feature_store.py",
            "validators.py"
        ],
        "models": [
            "__init__.py",
            "supervised.py",
            "unsupervised.py",
            "neural_networks.py",
            "rule_engine.py",
            "ensemble.py"
        ],
        "guardrails": [
            "__init__.py",
            "data_guardrails.py",
            "model_guardrails.py",
            "ethical_guardrails.py",
            "decision_guardrails.py",
            "explainability.py",
            "advanced_explainability.py"
        ],
        "agent": [
            "__init__.py",
            "memory.py",
            "reasoner.py",
            "planner.py",
            "decision.py",
            "executor.py",
            "a2a.py",
            "agent_cla.py"
        ],
        "governance": [
            "__init__.py",
            "policies.py",
            "audit.py"
        ],
        "monitoring": [
            "__init__.py",
            "drift_detector.py",
            "metrics.py",
            "alerting.py"
        ],
        "pipelines": [
            "__init__.py",
            "training.py",
            "inference.py",
            "retraining.py"
        ],
        "api": [
            "__init__.py",
            "fraud_api.py"
        ],
        "tests": [
            "__init__.py",
            "conftest.py",
            "test_guardrails.py",
            "test_models.py",
            "test_api.py",
            "performance",
            "load_test.js"
        ],
        "prometheus": [
            "prometheus.yml",
            "alert_rules.yml"
        ],
        "grafana": [
            "datasources",
            "prometheus.yml",
            "dashboards",
            "fraud_detection_dashboard.json"
        ],
        "kubernetes": [
            "deployment.yaml",
            "service.yaml",
            "hpa.yaml"
        ],
        ".github/workflows": [
            "ci-cd.yml"
        ],
        "root_files": [
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.dev.yml",
            "requirements.txt",
            "Makefile",
            ".env",
            ".dockerignore",
            ".gitignore",
            ".pre-commit-config.yaml",
            "setup.py",
            "README.md",
            "DEPLOYMENT_GUIDE.md",
            "main.py"
        ]
    }
}

def create_structure(base_dir, layout):
    """Recursively create directories and files."""
    base = Path(base_dir)
    for key, items in layout.items():
        if key == "root_files":
            for f in items:
                file_path = base / f
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch(exist_ok=True)
        else:
            sub_base = base / key
            if isinstance(items, list):
                sub_base.mkdir(parents=True, exist_ok=True)
                for f in items:
                    (sub_base / f).touch(exist_ok=True)
            elif isinstance(items, dict):
                create_structure(sub_base, items)

def main():
    print("Creating Fraud Detection System structure...")
    create_structure(".", structure)
    print("✅ Project structure created successfully!")

if __name__ == "__main__":
    main()

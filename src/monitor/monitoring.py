# monitoring_system.py
import psutil
from datetime import datetime
import random
import pandas as pd

class MonitoringSystem:
    def __init__(self):
        self.drift_history = []
        self.performance_history = []

    # System metrics
    def get_system_metrics(self):
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "active_threads": len(psutil.Process().threads()),
            "timestamp": datetime.now().isoformat()
        }

    # Simulate drift detection
    def check_drift(self):
        drift_value = random.uniform(0, 1)
        drift_status = "No significant drift" if drift_value < 0.5 else "Potential drift detected"
        self.drift_history.append({"timestamp": datetime.now().isoformat(), "drift_score": drift_value})
        return {"drift_score": drift_value, "status": drift_status}

    # Simulate model performance report
    def model_performance(self):
        accuracy = round(random.uniform(0.85, 0.98), 2)
        f1 = round(random.uniform(0.8, 0.95), 2)
        self.performance_history.append({"timestamp": datetime.now().isoformat(), "accuracy": accuracy, "f1": f1})
        return {"accuracy": accuracy, "f1_score": f1}

    # Health check
    def system_health(self):
        return {
            "status": "ok",
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "timestamp": datetime.now().isoformat()
        }

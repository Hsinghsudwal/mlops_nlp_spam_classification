from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Status(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    # INFO = "info"
    WARNING = "warning"
    # ERROR = "error"

    # CRITICAL = "critical"
    SKIPPED = "skipped"
    IDLE = "idle"
    RUNNING = "running"
    RETRYING = "retrying"


class Alert_Severity(str, Enum):
    ALERT_INFO = "info"
    ALERT_WARNING = "warning"
    ALERT_CRITICAL = "critical"


class Decision(str, Enum):
    """Decisions."""

    PROCEED = "proceed"
    RETRY = "retry"
    STOP = "stop"
    INVESTIGATE = "investigate"
    PROMOTE = "promote_if_valid"
    RETRAIN = "retrain"
    ROLLBACK = "rollback"
    UNKNOWN = "unknown"


@dataclass
class Result:
    status: Status
    data: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    message: str = ""

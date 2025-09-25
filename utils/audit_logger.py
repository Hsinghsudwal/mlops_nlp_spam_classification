import os
import logging
import requests
from datetime import datetime
from typing import Optional

# Environment/config defaults
LOG_FILE = "outputs/audit.log"
WEBHOOK_URL = "AUDIT_WEBHOOK"  # e.g., Slack webhook
MODE = "local"  # 'local' or 'hook'

# Setup logger
logger = logging.getLogger("audit")
if not logger.hasHandlers():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def audit_log(msg: str, details: dict = None, level: str = "INFO"):

    details = details or {}
    timestamp = datetime.utcnow().isoformat()
    message = f"[{msg}] {timestamp} {details}"

    # Log locally with caller info (stacklevel=2)
    getattr(logger, level.lower())(message, stacklevel=2)

    # Send webhook if hybrid mode
    if MODE == "hook" and WEBHOOK_URL:
        payload = {
            "msg": msg,
            "timestamp": timestamp,
            "details": details,
            "level": level,
        }
        try:
            requests.post(WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Webhook failed: {e}", stacklevel=2)

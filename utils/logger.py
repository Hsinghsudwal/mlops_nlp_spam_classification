import os
import logging
from datetime import datetime


class Logger:
    """Logger implementation for the MLOps."""

    def __init__(self, name: str = "SPAM", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)

        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # formatter = logging.Formatter(
        #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # )

        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create logs directory if it doesn't exist
        os.makedirs("outputs/logs", exist_ok=True)

        # Add file handler
        file_handler = logging.FileHandler(
            # f"outputs/logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            f"outputs/logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message: str):
        self.logger.info(message, stacklevel=2)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message, stacklevel=2)

    def debug(self, message: str):
        self.logger.debug(message)


# Create a singleton logger instance
logger = Logger()

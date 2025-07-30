import os
import logging
from datetime import datetime


class Logger:
    """Simple logger implementation for the MLOps."""

    def __init__(self, name: str = "SPAM", log_level: str = "INFO"):
        """
        Initialize the logger.

        Args:
            name: Name of the logger
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)

        # Set log level
        level = getattr(logging, log_level.upper())
        self.logger.setLevel(level)
        
        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create logs directory if it doesn't exist
        os.makedirs("outputs/logs", exist_ok=True)

        # Add file handler
        file_handler = logging.FileHandler(
            f"outputs/logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # self.logger.info("Logger initialized")

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)


# Create a singleton logger instance
logger = Logger()

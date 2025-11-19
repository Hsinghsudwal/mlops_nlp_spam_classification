import os
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split

from utils.logger import logger
from utils.audit_logger import audit_log
from utils.config_manager import ConfigManager
from utils.artifact_manager import ArtifactManager
from utils.results import Result, Status

class DataIngestion:
    """Data loading and split task."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self, file_path: str, column_names, separator) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == ".csv":
                return pd.read_csv(file_path)

            elif ext == ".tsv":
                return pd.read_csv(file_path, delimiter="\t", names=column_names)

            elif ext == ".parquet":
                return pd.read_parquet(file_path)

            elif ext == "":  # No extension, use separator
                return pd.read_csv(file_path, delimiter=separator, names=column_names)

            else:
                raise ValueError(f"Unsupported file format: {file_path}")

        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def validate_data(
        self, df: pd.DataFrame, text_column: str, target_column: str
    ) -> Tuple[bool, Optional[str]]:
        if df.empty:
            return False, "DataFrame is empty"

        if len(df) < 2:
            return False, "Dataset must contain at least 2 samples for train/test split"

        if target_column and target_column not in df.columns:
            return False, f"Missing target column: {target_column}"

        return True, None

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        return train_df, test_df

    def data_ingestion(
        self,
        results: Dict[str, Any],
        config: ConfigManager,
        artifact_store: ArtifactManager,
        pipeline_id: str,
    ) -> Result:
        try:

            data_path = self.data_path

            text_column = config.get("base.text_column", "text")
            target_column = config.get("base.target_column", "label")
            column_names = config.get("base.column_names", {})
            separator = config.get("base.separator", ",")
            test_size = config.get("base.test_size", 0.2)
            random_state = config.get("base.random_state", 42)
            raw_dir = config.get("dir_path.raw_path", "raw")

            # Load and validate data
            df = self.load_data(data_path, column_names, separator)

            is_valid, validation_error = self.validate_data(
                df, text_column, target_column
            )
            if not is_valid:
                raise ValueError(f"Data validation failed: {validation_error}")

            # Split data
            train_df, test_df = self.split_data(df, test_size, random_state)

            # Save artifacts
            artifact_store.save(train_df, raw_dir, "train.csv", pipeline_id)
            artifact_store.save(test_df, raw_dir, "test.csv", pipeline_id)
            audit_log("data ingestion", details={df.shape})
            return Result(
                status=Status.SUCCESS,
                # data={"train": train_df, "test": test_df},
                data={"train": train_df, "test": test_df},
                message="Data ingestion completed successfully",
            )

        except Exception as e:
            error_msg = f"Data ingestion failed: {str(e)}"
            logger.error(error_msg)
            audit_log("data ingestion", level="ERROR")
            return Result(status=Status.FAILED, message=error_msg)

    def run(self, **kwargs) -> Result:
        return self.data_ingestion(
            results=kwargs.get("results", {}),
            config=kwargs.get("config"),
            artifact_store=kwargs.get("artifact_store"),
            pipeline_id=kwargs.get("pipeline_id", "default"),
        )

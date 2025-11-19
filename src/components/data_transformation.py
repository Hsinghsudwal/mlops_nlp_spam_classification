import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils.results import Result, Status
from utils.logger import logger
from utils.config_manager import ConfigManager
from utils.artifact_manager import ArtifactManager

# Download NLTK resources (quiet to suppress console spam)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)


class WrapperPipeline(BaseEstimator, TransformerMixin):
    """Wrapper for text preprocessing + optional label encoding."""

    def __init__(
        self, preprocessing_pipeline: Pipeline, label_encoder: LabelEncoder = None
    ):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.label_encoder = label_encoder or LabelEncoder()

    def fit(self, X, y=None):
        # Fit preprocessing
        self.preprocessing_pipeline.fit(X)
        # Fit label encoder only if labels provided
        if y is not None:
            self.label_encoder.fit(y)
        return self

    def transform(self, X):
        """Transform only features for model consumption."""
        return self.preprocessing_pipeline.transform(X)

    def fit_transform(self, X, y=None):
        X_vec = self.preprocessing_pipeline.fit_transform(X)
        if y is not None:
            self.label_encoder.fit(y)
        return X_vec

    def encode_labels(self, y):
        return self.label_encoder.transform(y)

    def decode_labels(self, y_enc):
        return self.label_encoder.inverse_transform(y_enc)

    def get_classes(self):
        return self.label_encoder.classes_


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer for text cleaning."""

    def __init__(self, stop_words=None, lemmatizer=None):
        self.stop_words = stop_words or set(stopwords.words("english"))
        self.lemmatizer = lemmatizer or WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for text in X:
            try:
                text = re.sub("[^a-zA-Z0-9]", " ", str(text)).lower()
                tokens = text.split()
                tokens = [w for w in tokens if w not in self.stop_words]
                tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
                processed.append(" ".join(tokens))
            except Exception as e:
                logger.error(f"Error preprocessing text: {e}")
                processed.append("")
        return processed


class DataTransformation:
    """Data transformation for NLP tasks with WrapperPipeline."""

    def __init__(self):
        self.wrapper_pipeline = None

    def build_pipeline(self) -> WrapperPipeline:
        """Create Preprocessor + TF-IDF vectorizer + LabelEncoder wrapper."""
        preprocessing_pipeline = Pipeline(
            [
                ("cleaner", TextPreprocessor()),
                (
                    "vectorizer",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
            ]
        )
        label_encoder = LabelEncoder()
        return WrapperPipeline(preprocessing_pipeline, label_encoder)

    def feature_transforming(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> Tuple[Any, Any, np.ndarray, np.ndarray, WrapperPipeline]:
        """Apply preprocessing and TF-IDF vectorization."""
        if train_data.empty or test_data.empty:
            raise ValueError("Input dataframes cannot be empty.")

        X_train_raw = train_data[text_column].fillna("").tolist()
        X_test_raw = test_data[text_column].fillna("").tolist()
        y_train_raw = train_data[label_column].values
        y_test_raw = test_data[label_column].values

        # Build pipeline wrapper
        self.wrapper_pipeline = self.build_pipeline()

        logger.info("Fitting and transforming data using WrapperPipeline...")

        X_train = self.wrapper_pipeline.fit_transform(X_train_raw, y_train_raw)
        X_test = self.wrapper_pipeline.transform(X_test_raw)

        # Encode labels
        y_train = self.wrapper_pipeline.encode_labels(y_train_raw)
        y_test = self.wrapper_pipeline.encode_labels(y_test_raw)

        return X_train, X_test, y_train, y_test, self.wrapper_pipeline

    def data_transformation(
        self,
        results: Dict[str, Any],
        config: ConfigManager,
        artifact_store: ArtifactManager,
        pipeline_id: str,
    ) -> Result:
        try:
            ingestion = results.get("ingeste")
            if not ingestion or ingestion.status != Status.SUCCESS:
                raise ValueError("Ingestion results missing or failed")

            train_data = ingestion.data["train"]
            test_data = ingestion.data["test"]

            text_column = config.get("base.text_column", "text")
            target_column = config.get("base.target_column", "label")
            out_dir = config.get("dir_path.transformed", "transformed")

            # Transform
            (
                X_train,
                X_test,
                y_train,
                y_test,
                self.wrapper_pipeline,
            ) = self.feature_transforming(
                train_data, test_data, text_column, target_column
            )

            logger.info(f"Data shape: {X_train.shape}")

            output = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "transformer_pipeline": self.wrapper_pipeline,
            }

            # Save artifacts
            artifact_store.save(output, out_dir, "transformed.pkl", pipeline_id)

            return Result(
                status=Status.SUCCESS,
                data=output,
                message="Data transformation completed successfully",
            )

        except Exception as e:
            error_msg = f"Data transformation failed: {str(e)}"
            logger.error(error_msg)
            return Result(status=Status.FAILED, message=error_msg)

    def run(self, **kwargs) -> Result:
        return self.data_transformation(
            results=kwargs.get("results", {}),
            config=kwargs.get("config"),
            artifact_store=kwargs.get("artifact_store"),
            pipeline_id=kwargs.get("pipeline_id", "default"),
        )

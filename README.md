#   MLOPS_NLP_SPAM_CLASSIFICATION

## Overview

## Table of Content
- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Development](#development)
- [Pipeline](#pipeline)
- [Orchestration](#orchestration)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Best Practices](#best-practices)
- [Next Step](#next-step)

## Problem Statement 
Mobile message is a way of communication among the people, and billions of mobile device users exchange numerous messages. However, such type of communication is insecure due to lack of proper message filtering mechanisms. One cause of such insecurity is spam and The spam detection is a big issue in mobile message communication due to which mobile message communication is insecure. In order to tackle this problem, an accurate and precise method is needed to detect the spam in mobile message communication. Our job is to create a model which predicts whether a given SMS is spam or ham.

## Orchestration
data_generator==>data_collection==>data_preprocessor==>model_training==>serving==>monitoring==>governance


## Installation
```bash
# Clone the repository
git clone https://github.com/your-org/mlops_nlp_spam_classification.git
cd mlops_nlp_spam_classification

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Quick Start

```bash
# Start the development environment
docker-compose up -d

# Run the data ingestion pipeline
python src/data/ingest.py

# Train a model
python src/models/train.py --config configs/model/default.yaml

# Start the prediction API
python src/api/main.py
```

## Development

### Project Flow:
1. Problem Statement
2. Data Gathering
3. Data Preprocessing : Here we perform some operation on data
    A. Tokenization
    B. Lower Case
    C. Stopwords 
    D. Lemmatization / Stemming
4. Vectorization (Convert Text data into the Vector):
    A. Bag Of Words (CountVectorizer)
    B. TF-IDF
5. Model Building :
    A. Model Object Initialization
    B. Train and Test Model
6. Model Evaluation :
    A. Accuracy Score
    B. Confusition Matrix
    C. Classification Report
7. Model Deployment
8. Prediction on Client Data

## Tech Stack Used
1. Python
2. NLP
3. Machine Learning Algorithms


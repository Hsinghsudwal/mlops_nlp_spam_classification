#   MLOPS_NLP_SPAM_CLASSIFICATION

## Overview

## Table of Content
- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Development](#development)
- [Components](#components)
- [Pipeline](#pipeline)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Best Practices](#best-practices)
- [Next Step](#next-step)

## Problem Statement 

Every day, people receive dozens of text messages—some from friends, some from services they use, and unfortunately, many from unknown senders trying to trick or annoy them. These unwanted messages, known as spam, often carry phishing links, fake offers, or other. Manually filtering them is exhausting, and missing a real message because it looked like spam can be frustrating or even dangerous. For mobile users and platforms alike, this creates a need for smart, automated systems that can tell the difference between helpful messages (ham) and harmful ones (spam).

### Objective
The goal is to build an intelligent SMS classifier that can automatically identify whether a message is spam and efficiently.
1. Understand and clean up SMS text.
2. Learn from patterns in spam vs. ham messages.
3. Predict the category of new, unseen messages.

### Solution
We’ll use machine learning to solve this. Here's the plan:

1. Data Understanding & Cleaning: We'll start by cleaning the SMS messages—removing noise, standardizing text, and transforming them into a format a computer can understand.
2. Model Training: We'll train a model (like Naive Bayes, SVM, or try BERT) on a dataset of labeled messages so it learns what spam looks like.
3. Evaluation: We'll test how well it performs using real metrics like accuracy and precision, making sure it doesn’t wrongly flag genuine messages.
4. Deployment: The model could then be deployed as a live service that checks messages in real time.
5. MLOps Integration: To make the system smarter and more reliable, we can add an agent that keeps an eye on how well the model is working. If it notices the model making more mistakes or the data changing over time, it can automatically retrain the model. This helps the system stay accurate and useful in the long run.


## Installation
```bash
# Clone the repository
git clone https://github.com/Hsinghsudwal/mlops_nlp_spam_classification.git
cd mlops_nlp_spam_classification

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up conda environment
conda create -n myenv python=3.10 -y
conda activate myenv
pip install -r requirements.txt 

#  UV create and activate environment
uv venv
.venv\Scripts\activate  # On Windows
uv pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration
```

## Development

**Notebook:** Run Jupyter Notebook: On the terminal, from your main project directory.

`cd notebook` and `jupyter lab`

Dataset:  EDA, Text preprocessing, Feature Engineering, Model Trainer and Model Evaluation.

Build various ML models that can predict classification. The models that are used:
    `MultinomialNB`
    `Support Vector Classifier (SVC)`
    `Decision Tree`
    `Random Forest`

## Components

**Src:** Components: On the terminal, from your main project directory.

`cd src/components`

### 1. Data Ingestion
- Loads data from source CSV file
- Splits into training and testing sets
- Saved datasets to `outputs_store/raw/`

### 2. Data Transformation
- Applies feature engineering and preprocessing
- Handles categorical variables, scaling, and encoding
- Creates feature matrices (X) and target vectors (y)
- Saves preprocessed data and transformer model to `outputs_store/transformer/`


data_generator==>data_collection==>data_preprocessor==>model_training==>serving==>monitoring==>governance
## Pipeline


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

## Tech Stack Used
1. Python
2. NLP
3. Machine Learning Algorithms


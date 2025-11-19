# MLOPS_NLP_SPAM_CLASSIFICATION

## Overview
This project implements an end-to-end **SMS Spam Classification** system using NLP and MLOps best practices. It ingests SMS data, trains ML models to classify messages as spam or ham, deploys the model as an API, and includes monitoring for drift and performance.

---
## Table of Content
- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Research Notebook](#research-notebook)
- [Components](#components)
- [Pipeline](#pipeline)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Best Practices](#best-practices)
- [Next Steps](#next-steps)

---

## Problem Statement
Every day, users receive numerous SMS messages—some legitimate and some spam. Spam messages can contain phishing links, scams, or malicious content. Manually filtering messages is inefficient, and misclassifying important messages can have consequences. 

**Objective:**  
Build an intelligent system to automatically classify SMS messages as **spam** or **ham**.

**Solution Approach:**
1. **Data Understanding & Cleaning:** Clean text, remove noise, and standardize messages.
2. **Feature Engineering:** Transform messages into numerical representations suitable for ML models.
3. **Model Training:** Train models such as Naive Bayes, SVM, Decision Trees, or even transformer-based models like BERT.
4. **Evaluation:** Validate performance with metrics like accuracy, F1-score, and precision.
5. **Deployment:** Serve the model via a Flask API for real-time predictions.
6. **MLOps Integration:** Monitor system performance, detect drift, and retrain models automatically when needed.

---

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

## Research Notebook

**Notebook:** Run Jupyter Notebook: On the terminal, from your main project directory.

```bash
cd notebook and jupyter lab
```
#### Contents:

Dataset:  Exploratory Data Analysis (EDA), Text Preprocessing & Cleaning, Feature Engineering, Model Trainer and Model Evaluation.

Comparing models: predict classification.
    - `MultinomialNB`
    - `Support Vector Classifier (SVC)`
    - `Decision Tree`
    - `Random Forest`

## Components

**Src:** Components: On the terminal, from your main project directory.

`cd src/components`

### 1. Data Ingestion
- Loads data from source CSV file
- Splits into training and testing sets
- Saved datasets to `outputs/raw/`

### 2. Data Transformation
- Applies feature engineering and preprocessing
- Handles categorical variables, scaling, and encoding
- Creates feature matrices (X) and target vectors (y)
- Saves preprocessed data and transformer model to `outputs/transformer/`

### 3. Model Training
- Train classification models
- Evaluate and save best-performing model
- Save model artifacts for inference to `outputs/train/`

### 4. Model Serving
- Flask API to serve predictions
- Supports single and batch message predictions

### 5. Governance & Monitoring
- Track model performance and drift
- Detect anomalies in prediction patterns
- Alert on performance degradation or data drift

#### Flow:
`data_generator → data_collection → data_preprocessor → model_training → serving → governance`


data_generator==>data_collection==>data_preprocessor==>model_training==>serving==>governance

## Pipeline


#### Quick Start

```bash
# Start the development environment
.venv/Script/activate

# Run the pipeline
uv run main.py --local --mode train

# Deploy a model
uv run main.py --local --mode deploy

# Start the prediction API
uv run main.py --local --mode serve

# monitoring a model
uv run main.py --local --mode monitor
```

## Deployment

1. Model served via Flask API
2. Dockerized for consistent deployment
3. Endpoint supports:
    - Single message prediction
    - Batch CSV prediction

## Monitoring
1. Placeholder need to build evidently report and dashboard
    - Drift detection using Evidently
    - System metrics (CPU, memory, disk usage)
    - Model performance metrics (accuracy, F1-score)
2. Setup Systems:
    - Alerts for retraining or rollback actions or feedback-loop
    - Dashboard shows metrics as interactive cards

## Best Practices
1. Version control for datasets and models
2. Maintain experiment logs
3. Automated testing of pipeline components
4. Continuous integration with Docker & CI/CD
5. Use `.env` and `config files` for environment management

## Next Steps
- Integrate transformer-based models like BERT for improved accuracy
- Implement advanced feature engineering (TF-IDF, embeddings)
- Add continuous model retraining on live data
- Expand monitoring with visual dashboards using Plotly/Grafana
- Deploy model on cloud (AWS/GCP/Azure) with MLOps pipeline

## Tech Stack Used
1. Python
2. NLP
3. Machine Learning Algorithms


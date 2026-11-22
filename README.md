# AI381 Lab 3 — MLOps: Experiment Tracking, Model Selection, CI/CD and Deployment

## Project Overview

This project demonstrates an end-to-end MLOps workflow, including machine learning experiment tracking, model evaluation, model selection, automated testing, CI/CD, containerization, and deployment.

The project includes:
- A Telco Customer Churn experiment for comparing candidate machine learning models.
- An Iris Classification API built with FastAPI for containerization and deployment.

## Technologies

- Python 3.12
- scikit-learn
- MLflow
- FastAPI
- pytest
- Ruff
- Docker
- Docker Compose
- GitHub Actions
- Trivy
- GitHub Container Registry (GHCR)

## Churn Model Experiment

The MLflow experiment is named `AI381-Lab3-Churn`.

Dataset:
`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Three candidate models were trained and evaluated:

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 0.7257 | 0.6069 |
| Random Forest | 0.7640 | 0.5921 |
| SVM | 0.7257 | 0.6004 |

### Model Selection

Logistic Regression was selected as the champion model based on its highest F1-score, lowest p95 prediction latency, and smallest model size.

The selected model was registered in MLflow as `ChurnChampion`, with version 1 assigned the `champion` alias.

### Promotion Gate

A promotion gate compares a challenger model against the current champion using
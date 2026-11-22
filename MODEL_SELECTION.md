# Model Selection Report

## Project
AI381 - MLOps Lab 3: Churn Prediction

## Objective
The objective is to train, track, evaluate, and compare three machine learning models using MLflow. The best-performing model is selected as the Champion based on predictive performance, latency, and model size.

## Experiment
- MLflow Experiment: `AI381-Lab3-Churn`
- Models evaluated: Logistic Regression, Random Forest, Support Vector Machine (SVM)
- Dataset: Telco Customer Churn
- Data Hash (SHA-256): `88be4b93fbe0cc83421f1c503794c97c342eca914c1576db7c276e61d61358a`

## Models Evaluated

| Model | Accuracy | F1-Score | Contract Slice F1 | p95 Latency (ms) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7257 | 0.6069 | 0.0000 | 57.94 | 0.0072 |
| Random Forest | 0.7640 | 0.5921 | 0.0000 | 155.53 | 29.0030 |
| Support Vector Machine (SVM) | 0.7257 | 0.6004 | 0.0000 | 68.80 | 1.1025 |

## Champion Model

**Logistic Regression** was selected as the Champion model.

It achieved the highest F1-score (0.6069) among the three evaluated models. It also demonstrated the lowest p95 prediction latency (57.94 ms) and the smallest model size (0.0072 MB).

Although Random Forest achieved the highest accuracy (0.7640), its F1-score was lower, its latency was significantly higher, and its model size was considerably larger.

## Selection Criteria

The models were compared using:

- Accuracy
- F1-score
- Contract subgroup F1-score (weakest slice)
- p95 prediction latency
- Model size

F1-score was prioritized for model selection, while latency and model size were considered important deployment requirements.

## Weakest Slice Analysis

All three models achieved a Contract subgroup F1-score of 0.0000.

This indicates a significant weakness in predictions for this subgroup and highlights an area requiring further investigation and model improvement.

## MLflow Model Registry

The selected model was registered in MLflow Model Registry under:

- Registered Model Name: `ChurnChampion`
- Model Version: `1`
- Model Alias: `champion`
- Champion Run ID: `36f1029432164638b139faa12e8186d2`

## Promotion Gate

A promotion gate was implemented to compare a challenger model against the current Champion.

The Random Forest challenger was intentionally evaluated against Logistic Regression. The promotion gate blocked the promotion because the challenger did not achieve the required F1-score improvement and had higher prediction latency.

This demonstrates that the promotion gate prevents weaker models from replacing the current Champion.

## Conclusion

Logistic Regression was selected as the Champion because it provided the best F1-score, lowest prediction latency, and smallest model size among the evaluated candidates.

The model was registered in MLflow Model Registry and assigned the `champion` alias.

The evaluation also identified the Contract subgroup as an important area for future improvement.

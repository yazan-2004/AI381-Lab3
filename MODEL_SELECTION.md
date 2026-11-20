# Model Selection Report

## Project
AI381 - MLOps Lab 3: Iris Classification

## Objective
The objective is to train, evaluate, and compare three machine learning models using MLflow, then select the best-performing model as the Champion.

## Models Evaluated

Three models were trained and evaluated:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Random Forest | 0.9000 | 0.9024 | 0.9000 | 0.8997 |
| Support Vector Machine (SVM) | 0.9667 | 0.9697 | 0.9667 | 0.9666 |

## Champion Model

**Support Vector Machine (SVM)** was selected as the Champion model.

It achieved the highest accuracy (0.9667) and the highest macro F1-score (0.9666) among the evaluated models.

## Selection Criteria

The models were compared using:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score

The macro F1-score and accuracy were considered important indicators of overall classification performance.

## Conclusion

The SVM model demonstrated the best overall performance and was selected as the Champion model. It was saved as `model.joblib` for use by the FastAPI application and Docker deployment.
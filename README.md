# Fraud Detection Capstone Project 

## Overview
This project focuses on building and evaluating a **machine learning model for fraud detection** in financial transaction data. Fraud detection is a **highly imbalanced classification problem**, where fraudulent transactions are rare but have high business impact.  
The primary objective is to **maximize fraud recall** while maintaining a **manageable false positive rate**, enabling effective triage for manual review or further automated workflows.

---

## Tech Stack & Libraries
- **Python 3**
- pandas, numpy, scikit-learn
- xgboost / lightgbm / catboost
- imbalanced-learn (SMOTE, class weighting)
- matplotlib, seaborn
- shap (for explainability)

---

## Dataset
- Number of transactions: 200,000  
- Fraudulent transactions: 10,057 (5%)  
- **Key feature categories:**
  - Temporal features (Transaction Hour, Hour Bin, tx hour)
  - Account features (New Account, Account Age Days)
  - Transaction amount & log features
  - Geolocation/IP features

---

## Modeling Approach
1. **Data preprocessing**: handling missing values, scaling, encoding, and feature engineering.  
2. **Imbalance handling**: used SMOTE and class weights to address fraud rarity.  
3. **Model selection**: compared Logistic Regression, Random Forest, XGBoost and CatBoost models. 
4. **Hyperparameter tuning**: Hypertuned top 3 models using RandomizedSearchCV 
4. **Evaluation**: prioritized **precision-recall** performance over accuracy, due to the nature of fraud detection.

---

## Model Evaluation (Test Set)
| Metric                | Score       |
|------------------------|------------|
| Accuracy              | 0.909      |
| Precision (fraud=1)   | 0.32       |
| Recall (fraud=1)      | 0.70       |
| F1 Score (fraud=1)    | 0.44       |
| ROC-AUC               | 0.81       |
| PR-AUC                | 0.243      |

### Confusion Matrix

<img width="419" height="408" alt="Screenshot 2025-10-18 202754" src="https://github.com/user-attachments/assets/c91d0d83-8fd1-47a2-8a41-008ef06c7de2" />
```

- **High recall** indicates the model successfully detects a majority of fraudulent transactions.
- **Moderate precision** is acceptable for triage settings (manual review queues).
- **PR AUC of 0.24** is realistic for highly imbalanced fraud datasets.

---

## Feature Importance (Top Drivers)
1. Transaction Hour  
2. Hour Bin  
3. tx hour  
4. New Account  
5. Transaction Amount  
6. Account Age Days  

<img width="980" height="808" alt="Screenshot 2025-10-18 203008" src="https://github.com/user-attachments/assets/99cbbd59-41b2-4905-86f0-bfa803a8affd" />


<img width="1106" height="705" alt="Screenshot 2025-10-18 202924" src="https://github.com/user-attachments/assets/c83b7541-95ab-4400-b914-5780df90adca" />





These temporal and account-based signals are **key behavioral indicators** for fraud detection.

---

## Deployment Considerations
- **Use-case fit:** suitable for **triage** (flagging suspicious transactions) but **not** for auto-decline in production.  
- **Threshold tuning:** recommended to optimize precision vs. recall depending on operational cost.  
- **Monitoring:** add drift detection (PSI, KS, AUC tracking) and calibration checks.  
- **Pipeline:** wrap preprocessing + model in a single pipeline for inference.  
- **Shadow deployment** recommended before production rollout.

---

## Next Steps / Future Work
- Threshold tuning for business cost optimization  
- Temporal validation (rolling windows / backtests)  
- Probability calibration (isotonic / Platt scaling)  
- Cost-sensitive learning / focal loss  
- Add MLflow/DVC for experiment tracking  
- Deploy inference API for real-time scoring

---

## Notes
- This project demonstrates **realistic handling of an imbalanced classification problem** with **explainability and operational considerations**.  
- Suitable for academic submission, portfolio showcasing, or as a foundation for a production-grade fraud detection pipeline.

---

## Author
Developed by Opeyemi Aina

---

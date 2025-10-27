# Fraud Detection System

An end-to-end machine learning system for detecting fraudulent transactions using structured model pipelines, explainability tools, and production deployment components.

## Overview
This project focuses on building and evaluating a **machine learning model for fraud detection** in financial transaction data. Fraud detection is a **highly imbalanced classification problem**, where fraudulent transactions are rare but have high business impact.  
The primary objective is to **maximize fraud recall** while maintaining a **manageable false positive rate**, enabling effective triage for manual review or further automated workflows.

It follows a **two-stage approach**:

1. **Experimentation Phase (Notebook)**  
   - Exploratory Data Analysis (EDA)  
   - Feature engineering and model experimentation (Logistic Regression/Random forest/xgboost/catboost) 
   - Hyperparameter tuning on a 200,000-row dataset to select the best performing model.
   - **Evaluation**: prioritized **precision-recall** performance over accuracy, due to the nature of fraud detection.
   - Model explainability using feature importance and SHAP values 

2. **Production Phase (Pipeline)**  
   - Training the best model on the **entire dataset**  
   - Evaluating it on a hold-out test set  
   - Packaging and deploying it with streamlit

The final model(Catboost) achieves strong performance on fraud detection, with results shown below.

---

##  Demo

This project is deployed using streamlit providing an easy-to-use and interactive web interface for fraud detection.

## Live Demo / Recording

You can watch a short recording of the deployed app here: 
[Screenshot](<img width="1846" height="1008" alt="Screenshot 2025-10-27 114137" src="https://github.com/user-attachments/assets/581aa231-aeca-401c-9150-df8e73d8d7ea" />)

[View Deployment Recording](https://1drv.ms/v/c/50c5cfe66d856efa/EY0_F3F8HpNBp6dDOsTVY4MBED2x3eKlHUVROMagdqO20Q?e=0hrhHU)

# Dataset

The dataset used for training and testing the fraud detection model was sourced from Kaggle
.
It contains anonymized e-commerce transaction records with labels indicating whether each transaction is fraudulent (1) or legitimate (0).

- [Train data](https://www.kaggle.com/code/mdshafiuddinshajib/fraud-detection-ecommerce-transaction/input?select=Fraudulent_E-Commerce_Transaction_Data.csv)

- [Test data](https://www.kaggle.com/code/mdshafiuddinshajib/fraud-detection-ecommerce-transaction/input?select=Fraudulent_E-Commerce_Transaction_Data_2.csv)

Class Distribution (Training Set)
- Label	Description	Count
- 0	Legitimate	1,399,114
- 1	Fraudulent	73,838

This clearly shows a highly imbalanced dataset, which is common in fraud detection scenarios.

- Handling Class Imbalance

To address the imbalance and improve the model’s ability to detect fraudulent transactions, different techniques were used depending on the algorithm:

# Model	Technique Used	Description
- Logistic Regression	Class Weight	Gave higher importance to the minority (fraudulent) class during training.
- Random Forest	Class Weight	Weighted samples by class to reduce bias toward the majority class.
- CatBoost	SPW (Sample Weighting / Oversampling)	Applied oversampling strategy to balance class distribution.
- XGBoost	SPW (Sample Weighting / Oversampling)	Oversampled minority class to help the model learn fraud patterns.

This combination of class weights and sample weighting (SPW) helped ensure that each model was optimized for high recall and precision on the minority class, which is critical in fraud detection.

The final model(Catboost) achieves strong performance on fraud detection, with results shown below.

## Model Performance

| Metric                | Score       |
|-------------------------|------------|
| ROC-AUC                 | **0.8639** |
| PR-AUC                  | 0.6476     |
| Recall (fraud)          | 0.6882     |
| Precision (fraud)       | 0.3454     |
| F1 (fraud)              | 0.4599     |
| F2 (fraud)              | 0.5742     |
| Accuracy (overall)      | 0.9164     |

**Confusion Matrix (Label: 0 = Non-fraud, 1 = Fraud)**

|             | Predicted 0 | Predicted 1 |
|-------------|-------------|-------------|
| Actual 0    | 20818       | 1594        |
| Actual 1    | 381         | 841         |


---

## Key Capabilities

- **Modular ML pipeline** for training, evaluation, and inference.
- **EDA and model explainability** (feature importance + SHAP) during experimentation.
- **Config-driven architecture** for reproducibility and maintainability.
- **Production-ready model** with serialization, clean interfaces, and a user-friendly Streamlit interface for serving predictions
- **Containerized deployment** with Docker.
- **Comprehensive evaluation logging** in structured JSON format.

---

## Configuration (config.yaml)

This project is driven by a single YAML config that defines artifact locations and I/O paths for each pipeline stage.

<img width="668" height="644" alt="image" src="https://github.com/user-attachments/assets/4c576e5e-a2b4-46ff-916e-23e1a9bed8ee" />




## Project Structure

```
.
├── app.py                    # streamlit UI for predictions
├── main.py                   # Training pipeline entry point
├── config/                   # Config files (paths, params)
├── src/
│   ├── components/           # Data processing & model training modules
│   ├── pipeline/             # Training & prediction pipelines
│   └── utils/                # Utility functions
├── artifacts/                # Serialized models, transformers
├── notebooks/                # EDA, feature importance, SHAP analysis
├── evaluation_results.json   # Final model metrics on test data
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

##  Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/Fraud-Detection-System.git
cd Fraud-Detection-System
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python main.py
```

This runs the full training pipeline, saves the model and transformer to `artifacts/`, and updates evaluation metrics.

### 5. Start the prediction API

```bash
python app.py
```

---

---

## Model Explainability

### Feature Importance
During experimentation, **feature importance** was computed to identify which features contribute most to fraud detection.  

<img width="1106" height="705" alt="Screenshot 2025-10-18 202924" src="https://github.com/user-attachments/assets/c83b7541-95ab-4400-b914-5780df90adca" />


### SHAP Values
**SHAP** (SHapley Additive exPlanations) was used to provide local and global explanations for predictions.

<img width="980" height="808" alt="Screenshot 2025-10-18 203008" src="https://github.com/user-attachments/assets/99cbbd59-41b2-4905-86f0-bfa803a8affd" />

---

## Author
Developed by Opeyemi Aina




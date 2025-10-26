from pathlib import Path
from fraud_detection.utils.common import save_json  
import numpy as np
from fraud_detection.components.data_transformation import DataTransformation
from fraud_detection.entity import DataTransformationConfig, ModelEvaluationConfig
import logging
import joblib 
import os
import json
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report)



class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig, data_transformer):
        self.config = config
        self.data_transformer = data_transformer

    def evaluate(self):
        logging.info("Preparing test split from DataTransformation.")
        _, test_arr = self.data_transformer.initiate_data_transformation_and_split()
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
        assert X_test.shape[0] == y_test.shape[0], "Mismatched X_test/y_test sizes."

        logging.info("Loading saved CatBoost model from %s", str(self.config.best_model_path))
        model = CatBoostClassifier()
        model.load_model(str(self.config.best_model_path))

        logging.info("Scoring test data with the saved model.")
        y_pred = np.asarray(model.predict(X_test)).ravel().astype(int)

        # Probabilities 
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            s = model.decision_function(X_test); smin, smax = s.min(), s.max()
            y_prob = (s - smin) / (smax - smin + 1e-12)

        # Metrics 
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall    = float(recall_score(y_test, y_pred, zero_division=0))
        f1        = float(f1_score(y_test, y_pred, zero_division=0))
        f2        = float(((1+4)*precision*recall)/(4*precision+recall)) if (precision+recall)>0 else 0.0

        # PR AUC (never null)
        try:
            if y_prob is None:
                pr_auc = 0.0
            else:
                y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
                pr_auc = float(average_precision_score(np.asarray(y_test, dtype=int).ravel(), y_prob))
        except Exception:
            pr_auc = 0.0

        # ROC AUC 
        try:
            roc_auc = float(roc_auc_score(np.asarray(y_test, dtype=int).ravel(), y_prob)) if y_prob is not None else 0.0
        except Exception:
            roc_auc = 0.0

        # Confusion Matrix
        cm = confusion_matrix(np.asarray(y_test, dtype=int).ravel(), y_pred, labels=[0, 1])
        # Ensure JSON
        cm_list = cm.tolist()
        if cm.size == 4:
            tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        else:
            tn = fp = fn = tp = None  

        # Classification Report
        cls_report = classification_report(
            np.asarray(y_test, dtype=int).ravel(),
            y_pred,
            output_dict=True,
            zero_division=0
        )

        metrics = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc, 
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "f2": f2,
            "confusion_matrix": { 
                "labels_order": [0, 1],
                "matrix": cm_list,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp
            },
            "classification_report": cls_report 
        }

        with open(self.config.save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logging.info("Evaluation results saved to %s", str(self.config.save_path))
        return self.config.save_path, metrics
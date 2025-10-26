from pathlib import Path
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BEST_PARAMS = {
    "border_count": 102,
    "depth": 7,
    "iterations": 648,
    "l2_leaf_reg": 0.20495107742294383,
    "learning_rate": 0.1204262037029521,
    "scale_pos_weight": 9.442828734775043,
    "subsample": 0.7468055921327309,
    "bootstrap_type": "Bernoulli",
}

class ModelTrainer:
    def __init__(self, config, data_transformer):
        self.config = config
        self.data_transformer = data_transformer

    

    def train(self):
        train_arr, test_arr = self.data_transformer.initiate_data_transformation_and_split()

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0]  == y_test.shape[0]


        logging.info("Initializing CatBoostClassifier with BEST PARAMS.")
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            **BEST_PARAMS,
        )

        logging.info("Fitting CatBoost model on training data.")
        model.fit(X_train, y_train)

        score = float(model.score(X_test, y_test))

        model_path: Path = self.config.model_save_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        logging.info("Model saved to %s", str(model_path))

        return model_path, score
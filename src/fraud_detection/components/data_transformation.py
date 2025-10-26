import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import logging
from fraud_detection.utils.common import save_object, first_octet
from fraud_detection.entity import DataTransformationConfig
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_clean_data(file_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(file_path, low_memory=False)


class DataTransformation:
    def __init__(self, config: "DataTransformationConfig"):
        self.config = config

    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        df_no_target = df.drop(columns=["Is Fraudulent"], errors="ignore")

        # Feature type detection
        numeric_features = df_no_target.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = df_no_target.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        leak_cols = {"Transaction ID", "Customer ID", "IP Address", "Shipping Address", "Billing Address", "Transaction Date"}
        categorical_features = [c for c in categorical_features if c not in leak_cols]

        transformers = []

        # Numeric pipeline
        if numeric_features:
            num_pipe = Pipeline([("scaler", StandardScaler(with_mean=False))])
            transformers.append(("num", num_pipe, numeric_features))

          # Categorical pipeline
        if categorical_features:
           try:
            cat_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True, 
                dtype=np.float32
            )
           except TypeError:
            cat_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,       
                dtype=np.float32
            )
            cat_pipe = Pipeline([("encoder", cat_encoder)])
            transformers.append(("cat", cat_pipe, categorical_features))

        if not transformers:
            raise ValueError("No numeric or categorical columns available to build a preprocessor.")

        preprocessor = ColumnTransformer(
           transformers=transformers,
           sparse_threshold=1.0 
        )
        return preprocessor

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # datetime features
        dt = pd.to_datetime(df["Transaction Date"], errors="coerce")
        df["tx_year"] = dt.dt.year
        df["tx_month"] = dt.dt.month
        df["tx_dow"] = dt.dt.dayofweek
        df["tx_hour"] = dt.dt.hour

        # amount features
        df["Is Refund"] = (df["Transaction Amount"] < 0).astype(int)
        df["Log Abs Transaction Amount"] = np.log1p(df["Transaction Amount"].abs())
        df["Amount Bin"] = pd.qcut(df["Transaction Amount"], q=4, labels=False, duplicates="drop")

        # time-of-day features
        hour = df["tx_hour"].where(df["tx_hour"].notna(), df.get("Transaction Hour"))
        hour = pd.to_numeric(hour, errors="coerce")
        hour = hour.where((hour >= 0) & (hour <= 23))
        df["Hour Bin"] = pd.cut(hour, bins=[0, 6, 12, 18, 24], right=False, include_lowest=True, labels=False)

        df["Day of Week"] = df["tx_dow"]
        df["Is Weekend"] = df["Day of Week"].isin([5, 6]).astype(int)

        # address check (normalized)
        bill = df.get("Billing Address", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.strip().str.lower()
        ship = df.get("Shipping Address", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.strip().str.lower()
        df["Address Mismatch"] = (ship != bill).astype(int)

        # customer behaviour
        qty = pd.to_numeric(df.get("Quantity"), errors="coerce").replace(0, np.nan)
        df["Amount per Item"] = df["Transaction Amount"] / qty
        df["Age Amount Interaction"] = df.get("Customer Age", 0) * df["Transaction Amount"]
        if "Account Age Days" in df.columns:
            df["Account Age Bin"] = pd.qcut(df["Account Age Days"], q=4, labels=False, duplicates="drop")

        # IP 
        if "IP Address" in df.columns:
            df["IP First Octet"] = df["IP Address"].apply(first_octet)

        # simple risk flags
        q95 = df["Transaction Amount"].quantile(0.95)
        df["High Value Transaction"] = (df["Transaction Amount"] > q95).astype(int)
        if "Account Age Days" in df.columns:
            df["New Account"] = (df["Account Age Days"] < 30).astype(int)

        # drop leaky/high-cardinality columns
        leak_cols = ["Transaction ID", "Customer ID", "IP Address", "Shipping Address", "Billing Address", "Transaction Date"]
        df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")

        return df

    def process_file(self, file_path: Union[str, Path]):
        df = load_and_clean_data(file_path)
        df = self.engineer_features(df)

        # Create aligned X and y
        y = df.pop("Is Fraudulent")
        X = df
        return X, y

    def initiate_data_transformation_and_split(self):
        # Load and feature engineer
        X_train, y_train = self.process_file(self.config.train_path)
        X_test,  y_test  = self.process_file(self.config.test_path)

        assert len(X_train) == len(y_train), "Train X/y length mismatch before transform."
        assert len(X_test)  == len(y_test),  "Test X/y length mismatch before transform."

        logging.info("Building preprocessing pipeline.")
        preprocessor = self.build_preprocessor(X_train)

        logging.info("Applying preprocessing pipeline.")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed  = preprocessor.transform(X_test)

        assert len(X_train_processed) == len(y_train), "Train X/y length mismatch after transform."
        assert len(X_test_processed)  == len(y_test),  "Test X/y length mismatch after transform."

        # save the preprocessor
        save_object(file_path=self.config.preprocessor, obj=preprocessor)

        train_arr = np.c_[X_train_processed,np.array(y_train)]
        test_arr = np.c_[X_test_processed,np.array(y_test)]

        return train_arr, test_arr
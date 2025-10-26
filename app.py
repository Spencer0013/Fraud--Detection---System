import os
import tempfile
import joblib
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

from fraud_detection.utils.common import read_yaml
from fraud_detection.constants import CONFIG_FILE_PATH
from fraud_detection.entity import DataIngestionConfig, DataTransformationConfig
from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_transformation import DataTransformation  

# UI
st.set_page_config(page_title="Fraud Detection System",layout="wide")
st.title("Fraud Detection")
st.markdown("Upload a CSV of transactions; the model will flag fraud (0 = legit, 1 = fraud).")

#Load config & pipeline components
@st.cache_resource(show_spinner=False)
def _load_cfg_and_components():
    cfg = read_yaml(CONFIG_FILE_PATH) 
    ingest_cfg = DataIngestionConfig(**cfg.data_ingestion)
    trans_cfg  = DataTransformationConfig(**cfg.data_transformation)
    ingestion  = DataIngestion(ingest_cfg)
    transform  = DataTransformation(trans_cfg)       
    return cfg, ingest_cfg, trans_cfg, ingestion, transform

config, ingest_cfg, trans_cfg, data_ingestion, data_transformation = _load_cfg_and_components()

@st.cache_resource(show_spinner=False)
def _load_preprocessor(preproc_path: str):

    return joblib.load(preproc_path)

@st.cache_resource(show_spinner=False)
def _load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


model_path = getattr(config.model_trainer, "model_save_path", None)
if not model_path:
    model_path = getattr(getattr(config, "model_evaluation", {}), "best_model_path", None)
model = _load_model(model_path)
preprocessor = _load_preprocessor(trans_cfg.preprocessor)

use_thresholding = True
thr = st.slider("Decision threshold (positive class probability)", 0.05, 0.95, 0.50, 0.05)

# File upload
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded is not None:
    try:
        # 1) Peek columns to decide date parsing
        cols_head = pd.read_csv(uploaded, nrows=0).columns
        uploaded.seek(0)
        parse_dates = ["Transaction Date"] if "Transaction Date" in cols_head else None

        # 2) Read + preview
        df_raw = pd.read_csv(uploaded, low_memory=False, parse_dates=parse_dates)
        st.subheader("Preview")
        st.dataframe(df_raw.head(), use_container_width=True)

        df_clean = data_ingestion.convert_data_types(df_raw.copy())

        tmp_dir = tempfile.mkdtemp(prefix="fraud_streamlit_")
        tmp_path = os.path.join(tmp_dir, "clean.csv")
        df_clean.to_csv(tmp_path, index=False)
        X, _ = data_transformation.process_file(tmp_path)

        X_proc = preprocessor.transform(X)

        # 7) Predict
        pos_proba = model.predict_proba(X_proc)[:, 1]
        preds = (pos_proba >= thr).astype(int)

        # 8) Attach predictions to display DF
        out = df_raw.copy()
        out["Fraud_Prob"] = pos_proba
        out["Prediction"] = preds

        st.subheader("Prediction Results")
        show_cols = [c for c in ["Transaction Date", "Transaction Amount", "Payment Method",
                                 "Product Category"] if c in out.columns] + ["Fraud_Prob", "Prediction"]
        st.dataframe(out[show_cols].head(30), use_container_width=True)

        # 9) Summary
        total = len(out)
        frauds = int(out["Prediction"].sum())
        st.markdown(f"** Summary:** {frauds} flagged out of {total} transactions at threshold **{thr:.2f}**.")

        # 10) Download
        st.download_button(
            "Download CSV with Predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f" Processing error: {e}")



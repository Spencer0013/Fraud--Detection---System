from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class DataIngestionConfig:
    train_path : Path
    source_train_path: Path
    source_test_path: Path
    test_path : Path
    root_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
      root_dir: Path
      train_path: Path
      test_path:Path
      train_data: Path
      test_data:Path
      preprocessor: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir : Path
    model_save_path : Path

@dataclass(frozen=True)
class ModelTunerConfig:
    root_dir : Path
    tuner_save_path: str
    param_dist: dict
    cv_folds: int
    scoring: str
    model_save_path: Path
    model_name : str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir : Path
    best_model_path: Path
    save_path : Path
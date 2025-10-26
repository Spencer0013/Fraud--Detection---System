import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath("src"))

from fraud_detection.utils.common import read_yaml, create_directories
from fraud_detection.constants import CONFIG_FILE_PATH
from fraud_detection.constants import *
from fraud_detection.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelTunerConfig, ModelEvaluationConfig
from fraud_detection.utils.common import create_directories, save_object



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
        root_dir = config.root_dir,
        source_train_path = config.source_train_path,
        source_test_path = config.source_test_path,
        train_path = config.train_path,
        test_path = config.test_path
        )

        return data_ingestion_config
    



    def get_data_transformation(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
        root_dir = config.root_dir,
        train_path = config.train_path,
        test_path = config.test_path,
        train_data = config.train_data,
        test_data = config.test_data,
        preprocessor = config.preprocessor
        )

        return data_transformation_config
    


    def get_model_trainer(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_save_path=Path(config.model_save_path),
        )

        return model_trainer_config
    


    def get_model_evaluation(self) -> ModelEvaluationConfig:
        
        config = self.config.model_trainer.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            best_model_path=config.best_model_path,
            save_path =config.save_path
                    )

        return model_evaluation_config
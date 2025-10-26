import sys
sys.path.append("src") 


from fraud_detection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from fraud_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
from fraud_detection.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from fraud_detection.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from fraud_detection.logging import logger

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f"stage {STAGE_NAME} started") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f"stage {STAGE_NAME} started") 
   data_transformation = DataTransformationPipeline()
   data_transformation.main()
   logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f"stage {STAGE_NAME} started ") 
   model_trainer = ModelTrainerPipeline()
   model_trainer.main()
   logger.info(f" stage {STAGE_NAME} completed")
except Exception as e:
        logger.exception(e)
        raise e







STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f"stage {STAGE_NAME} started") 
   evaluation = ModelEvaluationPipeline()
   evaluation.main()
   logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
        logger.exception(e)
        raise e
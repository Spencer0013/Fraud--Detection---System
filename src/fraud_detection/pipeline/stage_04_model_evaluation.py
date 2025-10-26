from fraud_detection.config.configuration import ConfigurationManager
from fraud_detection.components.model_evaluation import ModelEvaluator
from fraud_detection.components.data_transformation import DataTransformation



class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation()
        data_transformer = DataTransformation(config=data_transformation_config)
        model_evaluation_config = config.get_model_evaluation()
        model_evaluation= ModelEvaluator(config=model_evaluation_config, data_transformer=data_transformer)
        model_evaluation.evaluate()
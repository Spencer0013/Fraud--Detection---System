from fraud_detection.config.configuration import ConfigurationManager
from fraud_detection.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation_and_split()
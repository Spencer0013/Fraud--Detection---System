import pandas as pd
from fraud_detection.entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config =config
        self.df_train = None
        self.df_test = None

    

    def read_data(self):
        self.df_train = pd.read_csv(self.config.source_train_path,low_memory=False,parse_dates=['Transaction Date'])
        self.df_test = pd.read_csv(self.config.source_test_path,low_memory=False,parse_dates=['Transaction Date'])

    def convert_data_types(self, df:pd.DataFrame) -> pd.DataFrame:
        df['Transaction Amount'] = pd.to_numeric(df['Transaction Amount'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['Customer Age'] = pd.to_numeric(df['Customer Age'], errors='coerce')
        df['Account Age Days'] = pd.to_numeric(df['Account Age Days'], errors='coerce')
        df['Transaction Hour'] = pd.to_numeric(df.get('Transaction Hour'), errors='coerce')  # if present
        df['Is Fraudulent'] = pd.to_numeric(df['Is Fraudulent'], errors='coerce').fillna(0).astype(int)
        return df
    
    def process_and_save(self):

        self.read_data()

        self.df_train = self.convert_data_types(self.df_train)

        self.df_test = self.convert_data_types(self.df_test)
    

        self.df_train.to_csv(self.config.train_path,index=False)
        self.df_test.to_csv(self.config.test_path, index=False)
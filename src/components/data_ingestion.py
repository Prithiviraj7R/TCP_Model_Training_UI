## Read the data 

import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## class for inputs related to data ingestion
## decorator named dataclass
'''
This dataclass library provides decorators for class. It automatically generates several 
special methods such as __init__, __repr__, __eq__, and __hash__ based on the class 
attributes you define.
'''

@dataclass
class DataIngestionConfig:
    X_train_data_path: str = os.path.join('artifacts','X_train.csv')
    Y_train_data_path: str = os.path.join('artifacts','Y_train.csv')
    X_test_data_path: str = os.path.join('artifacts','X_test.csv')
    Y_test_data_path: str = os.path.join('artifacts','Y_test.csv')
    X_raw_data_path: str = os.path.join('artifacts','X_raw_data.csv')
    Y_raw_data_path: str = os.path.join('artifacts','Y_raw_data.csv')
    X_data_path: str = os.path.join('artifacts','X_data.xlsx')
    Y_data_path:str = os.path.join('artifacts','Y_data.xlsx')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self,val_split):
        logging.info("Data Ingestion has started")
        try:
            '''
            Should be later modified to be read from SQL database
            '''
            # X_df = pd.read_excel('notebook\data\CM_X.xlsx',sheet_name=None)
            # Y_df = pd.read_excel('notebook\data\CM_Y.xlsx',sheet_name=None)

            X_df = pd.read_excel(r'uploaded_data\temperature_data.xlsx',sheet_name=None)
            Y_df = pd.read_excel(r'uploaded_data\thermal_displacement_data.xlsx',sheet_name=None)

            X_df = pd.concat(X_df.values(), ignore_index=True)
            Y_df = pd.concat(Y_df.values(), ignore_index=True)

            logging.info('Read the datasets as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.X_train_data_path), exist_ok=True)
            
            X_df.to_csv(self.ingestion_config.X_raw_data_path,index=False,header=True)
            Y_df.to_csv(self.ingestion_config.Y_raw_data_path,index=False,header=True)

            logging.info("Train Test split initiated")

            X_train, X_test, Y_train, Y_test = train_test_split(X_df,Y_df,test_size=val_split,random_state=42)

            X_train.to_csv(self.ingestion_config.X_train_data_path, index=False, header=True)
            X_test.to_csv(self.ingestion_config.X_test_data_path, index=False, header=True)
            Y_train.to_csv(self.ingestion_config.Y_train_data_path, index=False, header=True)
            Y_test.to_csv(self.ingestion_config.Y_test_data_path, index=False, header=True)

            logging.info("Ingestion of Train and Test data is complete")

            return (
                self.ingestion_config.X_train_data_path,
                self.ingestion_config.Y_train_data_path,
                self.ingestion_config.X_test_data_path,
                self.ingestion_config.Y_test_data_path,
                self.ingestion_config.X_raw_data_path,
                self.ingestion_config.Y_raw_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_day_wise_data_ingestion(self):
        logging.info("Data Ingestion for continual learning has started")
        try:
            X_df = pd.read_excel(r'uploaded_data\temperature_data.xlsx',sheet_name=None)
            Y_df = pd.read_excel(r'uploaded_data\thermal_displacement_data.xlsx',sheet_name=None)

            num_days = len(X_df.keys())
            return num_days

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    obj = DataIngestion()
    X_train_path,Y_train_path,X_test_path,Y_test_path,_,_ = obj.initiate_data_ingestion(0.2)

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(X_train_path,Y_train_path,X_test_path,Y_test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_comparison(train_arr,test_arr))
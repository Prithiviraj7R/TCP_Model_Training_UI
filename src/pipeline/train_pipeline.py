import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestionConfig,DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.components.data_extraction import DataExtraction

class TrainPipeline:
    def __init__(self):
        pass

    def train_selected_model(self,model_name,val_split):
        logging.info("Training the selected model has started")
        try:
            obj = DataIngestion()
            X_train_path,Y_train_path,X_test_path,Y_test_path,_,_ = obj.initiate_data_ingestion(val_split)

            data_transformation = DataTransformation()
            train_arr,test_arr,_ = data_transformation.initiate_data_transformation(X_train_path,Y_train_path,X_test_path,Y_test_path)

            model_trainer = ModelTrainer()
            report = model_trainer.initiate_model_trainer(model_name,train_arr,test_arr)
            
            return report
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_all_models(self,val_split):
        logging.info("Model comparison has begun")
        try:
            obj = DataIngestion()
            X_train_path,Y_train_path,X_test_path,Y_test_path,_,_ = obj.initiate_data_ingestion(val_split)

            data_transformation = DataTransformation()
            train_arr,test_arr,_ = data_transformation.initiate_data_transformation(X_train_path,Y_train_path,X_test_path,Y_test_path)

            model_trainer = ModelTrainer()
            report = model_trainer.initiate_model_comparison(train_arr,test_arr)

            return report
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_dl_model(self,val_split):
        logging.info("Training Deep Learning model")
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_online_model(self,val_split):
        logging.info("Training Online Learning model")
        try:
            model_trainer = ModelTrainer()
            report = model_trainer.initiate_online_learning(val_split)
            return report
        
        except Exception as e:
            CustomException(e,sys)

    
    def train_LSTM_model(self,val_split):
        logging.info("LSTM training pipeline")
        try:
            data_ingestion = DataIngestion()
            X_train_path,Y_train_path,X_test_path,Y_test_path = data_ingestion.initiate_sequence_ingestion(val_split)
            
            data_transformation = DataTransformation()
            train_arr,test_arr,_ = data_transformation.initiate_data_transformation(X_train_path,Y_train_path,X_test_path,Y_test_path)

            data_extraction = DataExtraction()
            X_train,Y_train,X_test,Y_test = data_extraction.initiate_data_extraction(train_arr,test_arr)

            model_trainer = ModelTrainer()
            report = model_trainer.initiate_LSTM_training(X_train,Y_train,X_test,Y_test)

            logging.info("LSTM model training pipeline is complete!")

            return report

        except Exception as e:
            raise CustomException(e,sys)
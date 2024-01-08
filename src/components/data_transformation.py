## Transformation of data
'''
This code involves data cleaning, data transformation and scaling the features, thus
preparing the data for model training.
'''

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join('artifacts','preprocesser.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            predictor_columns = ['Ambient', 'Ref Temp on Bed', 'Spindle Rear', 'Coolantwall', 'Transfomerbed', 'Spindle Front']

            predictor_pipeline = Pipeline(
                steps = [
                    ("scaler",MinMaxScaler())
                ]
            )

            logging.info(f"Min Max Scaling of features: {predictor_columns}")

            preprocesser = ColumnTransformer(
                [
                    ("predictor_pipeline",predictor_pipeline,predictor_columns)
                ]
            )

            return preprocesser

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,X_train_path,Y_train_path,X_test_path,Y_test_path):
        
        try:
            X_train = pd.read_csv(X_train_path)
            Y_train = pd.read_csv(Y_train_path)
            X_test = pd.read_csv(X_test_path)
            Y_test = pd.read_csv(Y_test_path)

            logging.info("Reading the train and test dataset")
            logging.info("Obtaining the preprocesser object")

            preprocessing_object = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing data.")

            X_train_scaled = preprocessing_object.fit_transform(X_train)
            X_test_scaled = preprocessing_object.transform(X_test)

            train_arr = np.concatenate((X_train_scaled, Y_train), axis=1)
            test_arr = np.concatenate((X_test_scaled, Y_test), axis=1)

            save_object(
                file_path = self.data_transformation_config.preprocesser_obj_file_path,
                obj = preprocessing_object
            )

            logging.info("Saving the preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            
        



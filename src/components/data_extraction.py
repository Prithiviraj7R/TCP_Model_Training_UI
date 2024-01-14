import os
import sys


from src.exception import CustomException
from src.logger import logging
from src.utils import createXY

import pandas as pd

class DataExtraction:
    def __init__(self):
        pass

    def initiate_data_extraction(self,train_arr,test_arr):
        logging.info("Data sequence generation has started")
        try:
            history = 10

            X_train,Y_train = createXY(train_arr,history)
            X_test,Y_test = createXY(test_arr,history)

            logging.info("Sequential Data Extraction completed!")

            return (
                X_train,
                Y_train,
                X_test,
                Y_test
            )

        except Exception as e:
            raise CustomException(e,sys)
        

import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            preprocesser_obj = load_object(os.path.join('artifacts','preprocesser.pkl'))
            scaler = preprocesser_obj.named_transformers_['target_pipeline'].named_steps['scaler']

            y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
            y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

            train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
            test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "y_train": y_train,
                "y_test": y_test,
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred
            }

            return report

    except Exception as e:
        raise CustomException(e,sys)
    




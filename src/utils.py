import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras import layers,models,initializers

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
    

def evaluate_deep_learning(model_name,X_train,y_train,X_test,y_test):
    try:
        report = {}

        if model_name == "DNN (Deep Neural Networks)":
            input_shape = X_train.shape[1]
            model = models.Sequential([
                        layers.Dense(96, input_shape=(input_shape,), activation='tanh', kernel_initializer=initializers.GlorotNormal()),
                        layers.BatchNormalization(),
                        layers.Dropout(0.2),
                        layers.Dense(64, activation='tanh', kernel_initializer=initializers.GlorotNormal()),
                        layers.BatchNormalization(),
                        layers.Dropout(0.2),
                        layers.Dense(32, activation='tanh', kernel_initializer=initializers.GlorotNormal()),
                        layers.BatchNormalization(),
                        layers.Dropout(0.2),
                        layers.Dense(16, activation='tanh', kernel_initializer=initializers.GlorotNormal()),
                        layers.BatchNormalization(),
                        layers.Dropout(0.2),
                        layers.Dense(8, activation='tanh', kernel_initializer=initializers.GlorotNormal()),
                        layers.BatchNormalization(),
                        layers.Dropout(0.2),
                        layers.Dense(1, kernel_initializer=initializers.GlorotNormal())
                    ])
            
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

            model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2,verbose=0)

            y_train_pred = model.predict(X_train,verbose=0)
            y_test_pred = model.predict(X_test,verbose=0)

            train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
            test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "y_train": y_train,
                "y_test": y_test,
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred
            }

            save_object(
                file_path = os.path.join(os.path.join('artifacts','trained_models'),f"{model_name}_model.pkl"),
                obj=model
            )
        
        else:
            pass

        return report

    except Exception as e:
        raise CustomException(e,sys)
    




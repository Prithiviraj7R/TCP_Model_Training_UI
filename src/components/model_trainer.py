## Training the model

import sys
import os
from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models,evaluate_deep_learning,load_object
from src.utils import evaluate_online_learning

@dataclass
class ModelTrainerConfig:
    trained_model_folder_path = os.path.join('artifacts','trained_models')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_model_instance(self, model_name):
        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }

        ## update this hyperparameter later
        params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {"learning_rate": [0.1, 0.01, 0.05, 0.001]},
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }
        
        if model_name in models:
            return {model_name: models[model_name]}, {model_name: params[model_name]}


    def initiate_model_trainer(self,model_name,train_array,test_array):
        try:
            logging.info("Split predictor and target data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model,param = self.get_model_instance(model_name)

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=model,
                params=param,
            )

            logging.info("Model training has been completed")

            save_object(
                file_path = os.path.join(self.model_trainer_config.trained_model_folder_path,f"{model_name}_model.pkl"),
                obj=model[model_name]
            )

            logging.info("Model has been saved as a pickle file")

            return model_report
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_model_comparison(self,train_array,test_array):
        try:
            models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
                }

            params = {
                    "Decision Tree": {
                        "criterion": [
                            "squared_error",
                            "friedman_mse",
                            "absolute_error",
                            "poisson",
                        ],
                    },
                    "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                    "Gradient Boosting": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                    "Linear Regression": {},
                    "XGBRegressor": {"learning_rate": [0.1, 0.01, 0.05, 0.001]},
                    "CatBoosting Regressor": {
                        "depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "iterations": [30, 50, 100],
                    },
                    "AdaBoost Regressor": {
                        "learning_rate": [0.1, 0.01, 0.5, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                }
            
            logging.info("Model Comparison has begun")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            for model_name in list(models.keys()):
                save_object(
                file_path = os.path.join(self.model_trainer_config.trained_model_folder_path,f"{model_name}_model.pkl"),
                obj=models[model_name]
                )

            model_names = list(model_report.keys())
            train_rmse_values = [model_report[model_name]['train_rmse'] for model_name in model_names]
            test_rmse_values = [model_report[model_name]['test_rmse'] for model_name in model_names]
            train_r2_values = [model_report[model_name]['train_r2'] for model_name in model_names]
            test_r2_values = [model_report[model_name]['test_r2'] for model_name in model_names]

            table_data = {
                'Train RMSE': train_rmse_values,
                'Test RMSE': test_rmse_values,
                'Train R2': train_r2_values,
                'Test R2': test_r2_values
            }

            results = pd.DataFrame(table_data, index=model_names)
            return results

        except Exception as e:
            raise CustomException(e,sys)
            

    def initiate_dl_training(self,model_name,train_array,test_array):
        try:
            logging.info("Split predictor and target data in Deep Learning")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model_report = evaluate_deep_learning(model_name,X_train,y_train,X_test,y_test)

            logging.info("Model training has been completed")

            logging.info("Model has been saved as a pickle file")

            return model_report

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_online_learning(self,val_days):
        try:
            logging.info("Online Learning Training has started")

            model_report = evaluate_online_learning(val_days)

            return model_report
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    model_trainer = ModelTrainer()
    report = model_trainer.initiate_online_learning(2)
    print(report)
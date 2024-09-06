import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV, train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import(LinearRegression, Lasso, Ridge,ElasticNet)
from sklearn.ensemble import(RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_dir = os.path.join(artifact_folder, 'model_trainer')
    trained_model_path = os.path.join(model_trainer_dir, 'trained_model', "model.pkl")
    expected_adjusted_rsquare = 0.6
    model_config_file_path = os.path.join('config', 'model.yaml')
    
    
class VisibilityModel:
    def __init__(self, preprocessing_object: ColumnTransformer, trained_model_object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Entered predict method of srcTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(X)

            logging.info("Used the trained model to get predictions")

            return self.trained_model_object.predict(transformed_feature)
        

        except Exception as e:
            raise CustomException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    
class ModelTrainer:
    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

        self.utils = MainUtils()

        self.models =  {
                    'linear':LinearRegression(),
                    'Lasso':Lasso(alpha=1.0),
                    'ridge':Ridge(),
                    'random':RandomForestRegressor(random_state = 13),
                    'xg':XGBRegressor(random_state = 42),
                    'Gradient':GradientBoostingRegressor(random_state = 42),
                    'knn': KNeighborsRegressor(),
                    'regr' : AdaBoostRegressor(random_state=0)
                }
        
    def evaluation(x_train, x_test, y_train, y_test, models):
        try:
            results = {}
            for key,value in models.items():
                model = value.fit(x_train,y_train)
                y_predict_test = model.predict(x_test)
                y_pred_train = model.predict(x_train)
                
                r2_train = r2_score(y_true = y_train,y_pred = y_pred_train)
                
                r2 = r2_score(y_true = y_test,y_pred = y_predict_test)
                
                mse = mean_squared_error(y_true = y_test,y_pred = y_predict_test)
                mae = mean_absolute_error(y_true = y_test,y_pred = y_predict_test)
                results[key] = {
                    'r2_score_train': r2_train,
                    'r2_score_test': r2,
                    'RMSE': np.sqrt(mse),
                    'MAE': mae,
                    'y_pred_test':y_predict_test,
                    'y_pred_train':y_pred_train
                }
                print(key)
                print(f'r2_score_train : {r2_train}')
                print(f'r2_score_test : {r2}')
                print(f'RMSE : {np.sqrt(mse)}')
                print(f'MAE : {mae}')
                print("%%%%%%%%%%%")
            return results
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_best_model(self,
                    x_train: np.array,
                    y_train: np.array,
                    x_test: np.array,
                    y_test: np.array):
        try:
            model_report: dict = self.evaluation(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=self.models
                )
            # Selecting the best model based on the highest R2 score on the test set
            best_model_name = max(model_report, key=lambda x: model_report[x]['r2_score_test'])
            best_model_object = self.models[best_model_name]
            best_model_score = model_report[best_model_name]['r2_score_test']
            
            print(f"Best Model: {best_model_name} with R2 score on test: {best_model_score}")
            
            return best_model_object, best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
        
    def finetune_best_model(self,
                            best_model_object: object,
                            best_model_name,
                            X_train,
                            y_train,
                            ) -> object:

        try:

            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][
                    best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1)

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            print("best params are:", best_params)

            finetuned_model = best_model_object.set_params(**best_params)

            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_model_trainer(self,
                               x_train,
                               y_train,
                               x_test,
                               y_test,
                               preprocessor_path):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            logging.info(f"Extracting model config file path")

            preprocessor = self.utils.load_object(file_path=preprocessor_path)

            logging.info(f"Extracting model config file path")

            model_report: dict = self.evaluation(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test, models=self.models)
            
            logging.info(f"Finding the best model based on R2 score")
            best_model_score = max([report['r2_score_test'] for report in model_report.values()])  # Get the highest R2 score
            best_model_name = max(model_report, key=lambda x: model_report[x]['r2_score_test'])  # Get the corresponding model name
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")
            best_model = self.models[best_model_name]

            # Fine-tune the best model
            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )
            
            # Test the model on the test data
            y_pred = best_model.predict(x_test)
            final_r2_score = r2_score(y_test, y_pred)
            final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            final_mae = mean_absolute_error(y_test, y_pred)
            
            logging.info(f"Final metrics for best model ({best_model_name}) - R2: {final_r2_score}, RMSE: {final_rmse}, MAE: {final_mae}")

            # Raise exception if the R2 score is below a threshold (e.g., 0.5)
            if final_r2_score < 0.5:
                raise Exception("No best model found with an R2 score greater than the threshold 0.5")

            # Create a custom model object containing the preprocessor and trained model
            logging.info(f"Saving best model and preprocessor as a VisibilityModel object")
            custom_model = VisibilityModel(
                preprocessing_object=preprocessor,
                trained_model_object=best_model
            )
            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
            )

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=custom_model,
            )
            
            self.utils.upload_file(from_filename=self.model_trainer_config.trained_model_path,
                                   to_filename="model.pkl",
                                   bucket_name=AWS_S3_BUCKET_NAME)

            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys)
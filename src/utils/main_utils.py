import sys
from typing import Dict, Tuple
import os

import numpy as np
import pandas as pd
import pickle
import yaml
import boto3

from src.constant import *
from src.exception import CustomException
from src.logger import logging


class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        """
        Reads a YAML file and returns its content as a dictionary.
        """
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        """
        Reads the schema YAML file for the project.
        """
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        """
        Saves a Python object to a file using pickle.
        """
        logging.info("Entered the save_object method of MainUtils class")

        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info("Exited the save_object method of MainUtils class")

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def load_object(file_path: str) -> object:
        """
        Loads a Python object from a file using pickle.
        """
        logging.info("Entered the load_object method of MainUtils class")

        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)

            logging.info("Exited the load_object method of MainUtils class")

            return obj

        except Exception as e:
            raise CustomException(e, sys) from e
        
    @staticmethod
    def upload_file(from_filename, to_filename, bucket_name):
        """
        Uploads a file to an AWS S3 bucket.
        """
        try:
            s3_resource = boto3.resource("s3")
            s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def download_model(bucket_name, bucket_file_name, dest_file_name):
        """
        Downloads a model file from an AWS S3 bucket.
        """
        try:
            s3_client = boto3.client("s3")
            s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)

            return dest_file_name

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def remove_unwanted_spaces(data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes unwanted spaces from all string columns in a pandas DataFrame.
        """
        try:
            df_without_spaces = data.apply(
                lambda x: x.str.strip() if x.dtype == "object" else x)
            logging.info('Successfully removed unwanted spaces from DataFrame.')
            return df_without_spaces
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def identify_feature_types(dataframe: pd.DataFrame):
        """
        Identifies the categorical, continuous, and discrete features in the given DataFrame.
        """
        data_types = dataframe.dtypes

        categorical_features = []
        continuous_features = []
        discrete_features = []

        for column, dtype in dict(data_types).items():
            unique_values = dataframe[column].nunique()

            if dtype == 'object' or unique_values < 10:  # Categorical if object type or < 10 unique values
                categorical_features.append(column)
            elif dtype in [np.int64, np.float64]:  # Continuous or discrete if numeric type
                if unique_values > 20:  # Continuous if more than 20 unique values
                    continuous_features.append(column)
                else:
                    discrete_features.append(column)
            else:
                # Handle other data types if needed
                pass

        return categorical_features, continuous_features, discrete_features

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Loads the CSV data from the given file path into a pandas DataFrame.
        """
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            raise CustomException(e, sys)

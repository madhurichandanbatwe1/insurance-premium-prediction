import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    data_transformation_dir = os.path.join(artifact_folder, 'data_transformation')
    transformed_train_file_path = os.path.join(data_transformation_dir, 'train.npy')
    transformed_test_file_path = os.path.join(data_transformation_dir, 'test.npy')
    transformed_object_file_path = os.path.join(data_transformation_dir, 'preprocessing.pkl')
    
class DataTransformation:
    def __init__(self,
                 valid_data_dir):

        self.valid_data_dir = valid_data_dir

        self.data_transformation_config = DataTransformationConfig()

        self.utils = MainUtils()
        
        
        
    @staticmethod
    def get_merged_batch_data(valid_data_dir: str) -> pd.DataFrame:
        """
        Method Name :   get_merged_batch_data
        Description :   This method reads all the validated raw data from the valid_data_dir and returns a pandas DataFrame containing the merged data. 
        
        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            raw_files = os.listdir(valid_data_dir)
            csv_data = []
            for filename in raw_files:
                data = pd.read_csv(os.path.join(valid_data_dir, filename))
                csv_data.append(data)

            merged_data = pd.concat(csv_data)


            return merged_data
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self):
        """
            Method Name :   initiate_data_transformation
            Description :   This method initiates the data transformation component for the pipeline 
            
            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class"
        )
        
        try:
            dataframe = self.get_merged_batch_data(valid_data_dir=self.valid_data_dir)
            dataframe = self.utils.remove_unwanted_spaces(dataframe)
            dataframe.replace('?', np.NaN, inplace=True)  # replacing '?' with NaN values for imputation
            
            df_num=dataframe.select_dtypes(include=['int64','float64'])
            df_cat=dataframe.select_dtypes(include=['object'])
            
            onehotEnc = OneHotEncoder(sparse=False, dtype=int)

            # Fit and transform the categorical data
            df_cat_encoded = onehotEnc.fit_transform(df_cat)

            # Get the feature names for the one-hot encoded columns
            encoded_columns = onehotEnc.get_feature_names_out(df_cat.columns)

            # Create a DataFrame with the encoded data and the appropriate column names
            df_cat_encoded = pd.DataFrame(df_cat_encoded, columns=encoded_columns, dtype=int)
            
            df_combined=pd.concat([df_num,df_cat_encoded],axis=1)
            
            X=df_combined.drop(columns='expenses',axis=1)
            y=df_combined['expenses']
            
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            
            scaler=MinMaxScaler()
            X_train_scaled=scaler.fit_transform(X_train)
            X_test_scaled=scaler.transform(X_test)
            
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path,
                                   obj=scaler)

            return X_train_scaled, y_train, X_test_scaled, y_test, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys) from e
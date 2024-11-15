import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import customException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        """
        This function is responsible for  data transformation
        """
        try:
            numeric_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 
                'lunch', 'test_preparation_course'
            ]
            
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Using median for numerical imputation
                    ("scaler", StandardScaler())  # No issue centering numerical data
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing categorical values
                    ("one_hot_encoder", OneHotEncoder(sparse_output=True)),  # Ensure output is sparse
                    ("scaler", StandardScaler(with_mean=False))  # Set with_mean=False for sparse data
                ]
            )

            # Log columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numeric_columns}")

            # Column transformer that applies the pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise customException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test Data Completed")
            logging.info("Obtaining preprocessor object")

            # Getting the preprocessor
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"  # This is the target variable we want to predict
            numeric_columns = ['reading_score', 'writing_score']

            # Separating features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test = test_df[target_column_name]

            logging.info("Applying preprocessor object to training dataset and test dataset")

            # Fit the transformer on the training data and transform both training and test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  # Use transform on test set

            # Combining features with target labels to create final datasets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            logging.info("Saving preprocessing object")

            # Save the preprocessor object for later use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed data and the path where the preprocessor object is saved
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise customException(e, sys)

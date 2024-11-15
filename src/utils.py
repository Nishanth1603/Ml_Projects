import os 
import sys
import numpy  as np
import pandas as pd
import dill 
from sklearn.metrics import r2_score

from src.exception import customException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise customException(e,sys)

def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        # Iterate over the models dictionary and evaluate each model
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Calculate R-squared scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store only the test model score in the report for simplicity
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise customException(e, sys)

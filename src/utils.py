import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import customException

# Function to save an object (e.g., trained model) to a file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        # Ensure the directory exists, or create it
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customException(e, sys)

# Function to evaluate multiple models, potentially with hyperparameter tuning
# In src/utils.py or wherever evaluate_models is defined
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param.get(model_name, {})

            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                gs.fit(x_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise customException(e, sys)


# Function to load a saved object (e.g., trained model)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customException(e, sys)

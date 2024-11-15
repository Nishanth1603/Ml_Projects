import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data.")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Dictionary of models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XgBoost": XGBRegressor(),
                "Linear regression": LinearRegression(),
                "k-neighbors regressor": KNeighborsRegressor(),
                "Catboost Regressor": CatBoostRegressor(verbose=False),
                "Adaboost Regressor": AdaBoostRegressor(),
            }

            # Evaluating models and getting the evaluation report (only test scores)
            model_report = evaluate_models(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models
            )

            # Get the best model score and its corresponding model name
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            # If the best model's score is less than 0.6, raise an exception
            if best_model_score < 0.6:
                raise customException(f"Best model score is below threshold: {best_model_score}")

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            # Get the best model from the models dictionary
            best_model = models[best_model_name]

            # Saving the trained best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Predicting and calculating R² score for the test data
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise customException(e, sys)

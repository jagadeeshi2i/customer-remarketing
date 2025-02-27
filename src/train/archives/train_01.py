import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "lr": LogisticRegression(),
                "rf": RandomForestClassifier(),
                "knn": KNeighborsClassifier(),
                "dt": DecisionTreeClassifier(),
                "bag": BaggingClassifier(),
                "sgd": SGDClassifier(),
                "xgb": XGBClassifier(),
            }

            params = {
                "lr": {
                    "penalty": ["l2"],
                    "C": [0.3, 0.6, 0.7],
                    "solver": ["sag"],
                },
                "rf": {
                    "criterion": ["entropy"],
                    "min_samples_leaf": [80, 100],
                    "max_depth": [25, 27],
                    "min_samples_split": [3, 5],
                    "n_estimators": [60, 70],
                },
                "knn": {"n_neighbors": [16, 17, 18]},
                "dt": {"max_depth": [8, 10], "min_samples_leaf": [1, 3, 5, 7]},
                "bag": {"n_estimators": [10, 15, 20]},
                "sgd": {
                    "loss": ["log", "huber"],
                    "learning_rate": ["adaptive"],
                    "eta0": [0.001, 0.01, 0.1],
                    "penalty": ["l1", "l2", "elasticnet"],
                    "alpha": [0.1, 1, 5, 10],
                },
                "xgb": {
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.1],
                    "n_estimators": [50],
                    "gamma": [0, 0.1],
                },
            }
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.5:
                raise CustomException("No best model found")
            logging.info(
                f"Best found model on both training and testing dataset"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict_proba(X_test)
            auc = roc_auc_score(y_test, predicted[:, 1])

            return auc

        except Exception as e:
            raise CustomException(e, sys)

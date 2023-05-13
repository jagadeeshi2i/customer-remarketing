import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

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
import numpy as np

from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from keras.callbacks import EarlyStopping


import warnings

warnings.filterwarnings("ignore")

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


class ModelTrainer:
    def __init__(self, input_folder,  output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        Path(str(self.output_folder)).mkdir(parents=True, exist_ok=True)

        self.trained_model_file_path = os.path.join(self.output_folder, "model.bst")
        
        self.train_array = np.load(os.path.join(self.input_folder, "train.npy"))
        self.test_array = np.load(os.path.join(self.input_folder, "test.npy"))
        
    def initiate_model_trainer(self):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                self.train_array[:, :-1],
                self.train_array[:, -1],
                self.test_array[:, :-1],
                self.test_array[:, -1],
            )

            def create_nn(
                activation="relu", optimizer="adam", dropout_rate=0.2
            ):
                model = Sequential()
                model.add(
                    Dense(10, input_dim=43, activation=activation.lower())
                )
                model.add(Dense(16, activation="tanh"))
                model.add(Dense(1, activation="sigmoid"))
                model.compile(
                    loss="binary_crossentropy",
                    optimizer=optimizer,
                    metrics=["accuracy"],
                )
                return model

            models = {
                "lr": LogisticRegression(),
                "rf": RandomForestClassifier(),
                "knn": KNeighborsClassifier(),
                "dt": DecisionTreeClassifier(),
                "bag": BaggingClassifier(),
                "sgd": SGDClassifier(),
                "xgb": XGBClassifier(),
                "NN": KerasClassifier(
                    build_fn=create_nn,
                    activation="relu",
                    callbacks=[EarlyStopping(monitor="accuracy", patience=3)],
                ),
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
                "NN": {
                    "batch_size": [32, 64, 128, 256],
                    "epochs": [2, 4, 8],
                    "optimizer": ["adam", "rmsprop"],
                    "activation": ["relu", "sigmoid"],
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
            logging.info(f"Found model on both training and testing dataset")

            # save_object(
            #     file_path=self.trained_model_file_path,
            #     obj=best_model,
            # )

            best_model.save_model(self.trained_model_file_path)

            predicted = best_model.predict_proba(X_test)
            auc = roc_auc_score(y_test, predicted[:, 1])
            logging.info(
                "Best model is {} and its AUC value is {:.2f}".format(
                    best_model_name, auc
                )
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()
    modeltrainer = ModelTrainer(args.input_folder, args.output_folder)
    modeltrainer.initiate_model_trainer()

import os
import sys

import numpy as np
import pandas as pd
import dill
import yaml

import pickle
import joblib

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score


def save_object(file_path, obj):
    try:
        # Check if the file already exists
        if os.path.exists(file_path):
            # If it does, remove the file
            os.remove(file_path)

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        # Save the model as joblib file
        foo = file_path.split('.')
        foo[-1] = 'joblib'
        joblib_file_path = '.'.join(foo)
        
        with open(joblib_file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        
        bar = file_path.split('/')
        bar[-1] = 'metadata.yaml'
        metadata_file_path = '/'.join(bar)

        metadata = {
                        "name": "custome",
                        "versions": ["iris/v1"],
                        "platform": "sklearn",
                        "inputs": [
                            {
                                "datatype": "BYTES",
                                "name": "input",
                                "shape": [4]
                            }
                        ],
                        "outputs": [
                            {
                                "datatype": "BYTES",
                                "name": "output",
                                "shape": [3]
                            }
                        ]
                    }

        with open(metadata_file_path, "wb") as file_obj:
            yaml.dump(metadata, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict_proba(X_train)

            y_test_pred = model.predict_proba(X_test)

            train_model_score = roc_auc_score(y_train, y_train_pred[:, 1])

            test_model_score = roc_auc_score(y_test, y_test_pred[:, 1])

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

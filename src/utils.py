import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , "wb") as file_obj:
            dill.dump(obj , file_obj)
    except Exception as e:
        raise CustomException(e , sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models , param):
    """
    Evaluate the performance of different regression models and return a report.
    
    Parameters:
    - X_train: Training feature set
    - y_train: Training target values
    - X_test: Testing feature set
    - y_test: Testing target values
    - models: Dictionary of model names and their instances
    
    Returns:
    - model_report: Dictionary with model names as keys and their R2 scores as values
    """
    model_report = {}
    
    for model_name, model in models.items():
        params = param.get(model_name, {})

        gs = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=2)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2_square = r2_score(y_test, y_pred)
        model_report[model_name] = r2_square
    
    return model_report

def load_object(file_path):
    """
    Load an object from a file.
    
    Parameters:
    - file_path: Path to the file containing the object
    
    Returns:
    - The loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)